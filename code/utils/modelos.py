"""
Arquitecturas de las redes neuronales del proyecto:
- PhaseNet / EQTransformer (detección, vía SeisBench)
- TCN (Temporal Convolutional Network)
- GNN (GCN y GAT)
- Transformer geoespacial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import seisbench.models as sbm

def get_base_detector(model_name = "PhaseNet", pretrained = True):
    """
    Carga PhaseNet o EQTransformer desde SeisBench, opcionalmente con pesos preentrenados en STEAD.

    :param model_name: "PhaseNet" o "EQTransformer".
    :param pretrained: Si True, devuelve el modelo con pesos preentrenados en STEAD.
    """
    model_classes = {
        "PhaseNet": sbm.PhaseNet,
        "EQTransformer": sbm.EQTransformer,
    }

    if model_name not in model_classes:
        raise ValueError("Modelo no soportado. Elige 'PhaseNet' o 'EQTransformer'.")

    cls = model_classes[model_name]

    if pretrained:
        return cls.from_pretrained("stead")
    return cls()

class Chomp1d(nn.Module):
    """Auxiliar para hacer que las convoluciones en la TCN sean causales."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """Un bloque residual para la Temporal Convolutional Network (TCN)."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """
    Temporal Convolutional Network que procesa secuencias de características temporales.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNClassifier(nn.Module):
    """
    TCN + cabeza de clasificación binaria para detección de precursores sísmicos.
    Pooling agregado [last, mean, max] sobre la dimensión temporal: combina el
    estado final causal con estadísticos globales de la ventana. Esto da al
    modelo acceso explícito a patrones estadísticos (mean/max) y al estado
    "presente" (last), que es la combinación estándar para clasificación de
    series cortas con TCN.

    Con num_channels=(32, 32, 64, 64) y kernel_size=3, el receptive field es 61,
    cubriendo la ventana LOOKBACK=60 días.

    Si num_outputs=1, devuelve (batch,). Si num_outputs>1, devuelve (batch, num_outputs)
    para clasificación binaria multi-label (un sigmoide por output independiente).
    """
    def __init__(self, num_inputs, num_channels = (32, 32, 64, 64), kernel_size = 3,
                 dropout = 0.2, num_outputs = 1):
        super().__init__()
        self.tcn = TCN(num_inputs, list(num_channels), kernel_size = kernel_size, dropout = dropout)
        self.fc  = nn.Linear(3 * num_channels[-1], num_outputs)
        self.num_outputs = num_outputs

    def forward(self, x):
        # x: (batch, seq_len, n_features) → Conv1d espera (batch, n_features, seq_len)
        x   = x.permute(0, 2, 1)
        out = self.tcn(x)                          # (batch, C, seq_len)
        out_last = out[:, :, -1]                   # estado causal final
        out_mean = out.mean(dim = 2)               # promedio temporal
        out_max  = out.max(dim = 2).values         # máximo temporal
        out = torch.cat([out_last, out_mean, out_max], dim = 1)   # (batch, 3C)
        out = self.fc(out)                         # (batch, num_outputs)
        if self.num_outputs == 1:
            return out.squeeze(-1)                 # (batch,)
        return out                                 # (batch, num_outputs)


class GCNLayer(nn.Module):
    """
    Capa de Graph Convolutional Network (Kipf & Welling, 2017):
        H' = σ(Â · H · W)
    donde Â = D̃^(-1/2) (A + I) D̃^(-1/2) es la matriz de adyacencia simétricamente
    normalizada con self-loops, pre-computada una sola vez fuera del modelo y
    pasada al forward (no es un parámetro entrenable).
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, A_norm):
        # x: (batch, n_nodes, in_features)
        # A_norm: (n_nodes, n_nodes), pre-normalizada
        x = self.linear(x)
        # Â · H: contracción sobre la dimensión de nodos (m → n)
        return torch.einsum("nm,bmf->bnf", A_norm, x)


class SpatioTemporalGNN(nn.Module):
    """
    Modelo espacio-temporal para predicción sísmica precursora:
      1. TCN compartido procesa la serie temporal de CADA celda independientemente
         (weight sharing entre celdas). Salida: embedding por celda.
      2. 2 capas GCN propagan información entre celdas vecinas vía el grafo.
      3. Cabeza lineal por nodo predice el logit binario (¿mainshock M≥4.5
         en celda c o vecinas en próximos T días?).

    Input:  (batch, n_nodes, lookback, n_features)
    Output: (batch, n_nodes) logits
    """
    def __init__(self, n_inputs, num_channels = (32, 32, 64),
                 kernel_size = 3, hidden_gnn = 64, dropout = 0.15):
        super().__init__()
        self.tcn         = TCN(n_inputs, list(num_channels),
                               kernel_size = kernel_size, dropout = dropout)
        self.gcn1        = GCNLayer(3 * num_channels[-1], hidden_gnn)
        self.gcn2        = GCNLayer(hidden_gnn,        hidden_gnn)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(hidden_gnn, 1)
        self._n_features = n_inputs

    def forward(self, x, A_norm):
        # x: (batch, n_nodes, lookback, n_features)
        # A_norm: (n_nodes, n_nodes), pre-normalizada (constante)
        B, N, T, F_in = x.shape

        # TCN aplicado a cada celda (batch · nodos como batch efectivo)
        h = x.reshape(B * N, T, F_in).permute(0, 2, 1)   # (B*N, F_in, T)
        h = self.tcn(h)                                  # (B*N, C_out, T)
        h_last = h[:, :, -1]                             # último timestep (causal)
        h_mean = h.mean(dim = 2)                         # promedio temporal
        h_max  = h.max(dim = 2).values                   # máximo temporal
        h = torch.cat([h_last, h_mean, h_max], dim = 1)  # (B*N, 3 * C_out)
        h = h.reshape(B, N, -1)                          # (B, N, 3 * C_out)

        # 2 capas GCN sobre el grafo
        h = F.relu(self.gcn1(h, A_norm))
        h = self.dropout(h)
        h = F.relu(self.gcn2(h, A_norm))
        h = self.dropout(h)

        return self.fc(h).squeeze(-1)                    # (B, N) logits


def normalize_adjacency(adj):
    """
    Normalización simétrica de la matriz de adyacencia (Kipf & Welling 2017):
        Â = D̃^(-1/2) (A + I) D̃^(-1/2)
    Devuelve un tensor torch float32 listo para usar en GCNLayer.forward.
    """
    import numpy as np
    A_tilde = adj + np.eye(adj.shape[0], dtype = adj.dtype)
    D       = A_tilde.sum(axis = 1)
    D_inv   = 1.0 / np.sqrt(D + 1e-8)
    A_norm  = (A_tilde * D_inv[:, None]) * D_inv[None, :]
    return torch.from_numpy(A_norm.astype(np.float32))


def adjacency_mask(adj):
    """
    Máscara binaria (N, N) con self-loops para GAT: vale 1 si el nodo i puede
    atender al nodo j (existe arista i-j o i==j), 0 en caso contrario. La capa
    GAT usa esta máscara para restringir la atención a los vecinos del grafo.
    """
    import numpy as np
    A = (adj + np.eye(adj.shape[0], dtype = adj.dtype)) > 0
    return torch.from_numpy(A.astype(np.float32))


class GATLayer(nn.Module):
    """
    Capa Graph Attention Network (Veličković et al., 2018), implementación densa.

    A diferencia de GCN (que pondera a los vecinos por el grado, de forma fija),
    GAT aprende coeficientes de atención α_ij sobre cada vecino, permitiendo que
    la red decida qué vecinas importan más (p.ej. celdas alineadas con la falla)
    e ignore las irrelevantes:

        e_ij = LeakyReLU(aᵀ [W h_i ‖ W h_j])
        α_ij = softmax_j(e_ij)        (sólo sobre los vecinos j de i)
        h_i' = Σ_j α_ij (W h_j)

    Soporta múltiples cabezas de atención. Con concat=True las cabezas se
    concatenan (salida H·out_features); con concat=False se promedian (out_features).
    """
    def __init__(self, in_features, out_features, n_heads = 1, dropout = 0.15,
                 concat = True, leaky_slope = 0.2):
        super().__init__()
        self.n_heads      = n_heads
        self.out_features = out_features
        self.concat       = concat
        self.W      = nn.Parameter(torch.empty(n_heads, in_features, out_features))
        self.a_src  = nn.Parameter(torch.empty(n_heads, out_features))
        self.a_dst  = nn.Parameter(torch.empty(n_heads, out_features))
        self.leaky  = nn.LeakyReLU(leaky_slope)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for h in range(self.n_heads):
            nn.init.xavier_uniform_(self.W[h])
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, mask):
        # x: (B, N, in_features) ; mask: (N, N) binaria (1 si arista i-j permitida)
        B, N, _ = x.shape
        H, Fout = self.n_heads, self.out_features

        # Proyección por cabeza: Wh (B, H, N, Fout)
        Wh = torch.einsum("bnf,hfo->bhno", x, self.W)

        # Scores de atención source/destination: (B, H, N)
        s_src = torch.einsum("bhno,ho->bhn", Wh, self.a_src)
        s_dst = torch.einsum("bhno,ho->bhn", Wh, self.a_dst)

        # e_ij = LeakyReLU(s_src_i + s_dst_j) → (B, H, N, N)
        e = self.leaky(s_src.unsqueeze(-1) + s_dst.unsqueeze(-2))

        # Enmascarar pares sin arista con -inf antes del softmax
        no_edge = (mask == 0).unsqueeze(0).unsqueeze(0)   # (1, 1, N, N)
        e = e.masked_fill(no_edge, float("-inf"))

        alpha = torch.softmax(e, dim = -1)                 # softmax sobre vecinos j
        alpha = self.dropout(alpha)

        # h'_i = Σ_j α_ij Wh_j → (B, H, N, Fout)
        out = torch.einsum("bhnm,bhmo->bhno", alpha, Wh)

        if self.concat:
            return out.permute(0, 2, 1, 3).reshape(B, N, H * Fout)   # (B, N, H·Fout)
        return out.mean(dim = 1)                                     # (B, N, Fout)


class SpatioTemporalGAT(nn.Module):
    """
    Variante de SpatioTemporalGNN que sustituye las 2 capas GCN por 2 capas GAT
    (atención sobre grafos). El bloque temporal (TCN compartido por celda) es
    idéntico; sólo cambia el mecanismo de propagación espacial.

      1. TCN compartido → embedding por celda.
      2. GAT capa 1 (multi-cabeza, concat) → hidden_gnn.
      3. GAT capa 2 (1 cabeza, promedio)   → hidden_gnn.
      4. Cabeza lineal por nodo → logit binario.

    Input:  (batch, n_nodes, lookback, n_features)
    Output: (batch, n_nodes) logits
    Segundo argumento del forward: máscara binaria de adyacencia (no la normalizada).
    """
    def __init__(self, n_inputs, num_channels = (32, 32, 64), kernel_size = 3,
                 hidden_gnn = 64, dropout = 0.15, n_heads = 4):
        super().__init__()
        assert hidden_gnn % n_heads == 0, "hidden_gnn debe ser divisible por n_heads"
        self.tcn     = TCN(n_inputs, list(num_channels),
                           kernel_size = kernel_size, dropout = dropout)
        self.gat1    = GATLayer(num_channels[-1], hidden_gnn // n_heads,
                                n_heads = n_heads, dropout = dropout, concat = True)
        self.gat2    = GATLayer(hidden_gnn, hidden_gnn,
                                n_heads = 1, dropout = dropout, concat = False)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_gnn, 1)
        self._n_features = n_inputs

    def forward(self, x, mask):
        B, N, T, F_in = x.shape

        # TCN aplicado a cada celda (weight sharing)
        h = x.reshape(B * N, T, F_in).permute(0, 2, 1)   # (B*N, F_in, T)
        h = self.tcn(h)                                  # (B*N, C_out, T)
        h_last = h[:, :, -1]                             # último timestep (causal)
        h_mean = h.mean(dim = 2)                         # promedio temporal
        h_max  = h.max(dim = 2).values                   # máximo temporal
        h = torch.cat([h_last, h_mean, h_max], dim = 1)  # (B*N, 3 * C_out)
        h = h.reshape(B, N, -1)                          # (B, N, 3 * C_out)

        # 2 capas GAT sobre el grafo
        h = F.elu(self.gat1(h, mask))
        h = self.dropout(h)
        h = F.elu(self.gat2(h, mask))
        h = self.dropout(h)

        return self.fc(h).squeeze(-1)                    # (B, N) logits


def geospatial_encoding(coords, d_model):
    """
    Codificación posicional sinusoidal 2D a partir de coordenadas geográficas.

    Análogo a la positional encoding de Vaswani et al. (2017) pero con la latitud
    y la longitud (en grados) como "posiciones" continuas. Se reserva la mitad de
    las dimensiones para la latitud y la otra mitad para la longitud; dentro de
    cada mitad se alternan senos y cosenos a frecuencias geométricamente espaciadas.

    :param coords: array (N, 2) con (lat, lon) de cada celda activa.
    :param d_model: dimensión del embedding (debe ser múltiplo de 4).
    :return: tensor torch float32 (N, d_model) listo para sumar a los embeddings.
    """
    import numpy as np
    assert d_model % 4 == 0, "d_model debe ser múltiplo de 4 (mitad lat, mitad lon)"
    coords = np.asarray(coords, dtype = np.float32)
    N = coords.shape[0]
    d_half = d_model // 2                                  # dims por coordenada
    div = np.exp(np.arange(0, d_half, 2) * (-np.log(10000.0) / d_half))  # (d_half/2,)

    pe = np.zeros((N, d_model), dtype = np.float32)
    for c in (0, 1):                                       # 0 = lat, 1 = lon
        pos = coords[:, c:c + 1]                           # (N, 1)
        base = c * d_half
        pe[:, base + 0 : base + d_half : 2] = np.sin(pos * div)
        pe[:, base + 1 : base + d_half : 2] = np.cos(pos * div)
    return torch.from_numpy(pe)


class SpatioTemporalTransformer(nn.Module):
    """
    Modelo espacio-temporal con atención geoespacial (ablation de la GNN/GAT).

    Mantiene el MISMO front-end temporal (TCN compartido por celda) que la GNN, y
    sustituye la propagación por grafo por un TransformerEncoder con self-attention
    full-pairwise sobre las celdas. La estructura espacial NO se impone vía un grafo
    explícito: se inyecta mediante una codificación geoespacial sinusoidal de las
    coordenadas (lat, lon) y se deja que la atención aprenda las relaciones.

      1. TCN compartido → embedding por celda.
      2. + codificación geoespacial sinusoidal (lat, lon).
      3. TransformerEncoder (atención full-pairwise entre celdas).
      4. Cabeza lineal por celda → logit binario.

    Input:  (batch, n_nodes, lookback, n_features)
    Output: (batch, n_nodes) logits
    Segundo argumento del forward: codificación geoespacial pe (n_nodes, d_model),
    pre-computada con geospatial_encoding(coords, d_model).

    Si se pasa attn_mask (N, N) booleana en el constructor, la atención queda
    restringida: los pares marcados con True NO se atienden (p.ej. bloquear que
    celdas de California y Alaska se atiendan entre sí). attn_mask=None (por
    defecto) deja la atención full-pairwise global.
    """
    def __init__(self, n_inputs, num_channels = (32, 32, 64), kernel_size = 3,
                 d_model = 64, n_heads = 4, n_layers = 2, dropout = 0.15,
                 dim_feedforward = 128, attn_mask = None):
        super().__init__()
        assert d_model % 4 == 0, "d_model debe ser múltiplo de 4 (codificación geoespacial)"
        self.tcn        = TCN(n_inputs, list(num_channels),
                              kernel_size = kernel_size, dropout = dropout)
        self.input_proj = (nn.Linear(3 * num_channels[-1], d_model)
                           if 3 * num_channels[-1] != d_model else nn.Identity())
        enc_layer = nn.TransformerEncoderLayer(
            d_model = d_model, nhead = n_heads, dim_feedforward = dim_feedforward,
            dropout = dropout, batch_first = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers = n_layers)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(d_model, 1)
        self.d_model     = d_model
        self._n_features = n_inputs
        # Máscara de atención (constante). Buffer → se mueve con .to(device).
        if attn_mask is not None:
            self.register_buffer("attn_mask", attn_mask.bool())
        else:
            self.attn_mask = None

    def forward(self, x, pe):
        # x: (B, N, T, F_in) ; pe: (N, d_model) codificación geoespacial
        B, N, T, F_in = x.shape

        # TCN aplicado a cada celda (weight sharing)
        h = x.reshape(B * N, T, F_in).permute(0, 2, 1)   # (B*N, F_in, T)
        h = self.tcn(h)                                  # (B*N, C_out, T)
        h_last = h[:, :, -1]                             # último timestep (causal)
        h_mean = h.mean(dim = 2)                         # promedio temporal
        h_max  = h.max(dim = 2).values                   # máximo temporal
        h = torch.cat([h_last, h_mean, h_max], dim = 1)  # (B*N, 3 * C_out)
        h = h.reshape(B, N, -1)                          # (B, N, 3 * C_out)

        # Proyección a d_model + codificación geoespacial
        h = self.input_proj(h)                           # (B, N, d_model)
        h = h + pe.unsqueeze(0)                           # broadcast sobre el batch

        # Self-attention sobre las celdas (global, o restringida si hay attn_mask)
        h = self.transformer(h, mask = self.attn_mask)   # (B, N, d_model)
        h = self.dropout(h)

        return self.fc(h).squeeze(-1)                    # (B, N) logits


def region_attention_mask(region_arr):
    """
    Máscara de atención (N, N) booleana para SpatioTemporalTransformer.

    Vale True (= bloquear atención) en los pares de celdas que pertenecen a
    regiones distintas, y False (= permitir) dentro de la misma región. Así, en
    el escenario California+Alaska, una celda sólo atiende a celdas de su propia
    región (incluida ella misma), replicando la separación tectónica que la GNN
    imponía vía aristas intra-región.

    :param region_arr: array (N,) con el nombre de región de cada celda activa.
    :return: BoolTensor (N, N), True donde la atención está prohibida.
    """
    import numpy as np
    r = np.asarray(region_arr)
    same  = (r[:, None] == r[None, :])    # True si misma región
    block = ~same                          # True si distinta región → bloquear
    return torch.from_numpy(block)         # BoolTensor (N, N)


class GeospatialTransformer(nn.Module):
    """
    Transformer con Codificación Geoespacial.
    Aplica mecanismos de atención sobre la red de estaciones para priorizar sismómetros.
    """
    def __init__(self, embed_dim, num_heads, num_layers):
        super(GeospatialTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Capa final para predicción/clasificación basada en la atención global
        self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, x, pe):
        # x: (Seq_len/N_estaciones, Batch, embed_dim)
        # pe: (Seq_len/N_estaciones, Batch, embed_dim) - Positional Encoding (coordenadas geográficas)
        
        # Suma de la codificación posicional
        x = x + pe

        # Codificador Transformer
        out = self.transformer(x)

        # Promedio sobre la secuencia (estaciones)
        out = out.mean(dim=0)

        # Cabeza de clasificación / regresión de riesgo
        out = self.fc_out(out)
        return out

class UncertaintyGP(nn.Module):
    """
    Gaussian Process Layer (Placeholder).
    Añade estimación de incertidumbre Bayesiana a las predicciones.
    """
    def __init__(self):
        super(UncertaintyGP, self).__init__()
        # En una implementación completa se integraría GPyTorch
        pass
        
    def forward(self, x):
        # Devuelve media y varianza simplificadas
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True) + 1e-6
        return mean, var
