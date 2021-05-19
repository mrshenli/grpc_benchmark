
# figure 1
# BS = 32

# 1 GPU
# a. local pipeline fwd-bwd
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd

# 2 GPU
# a. local pipeline fwd-bwd
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 4 GPU
# a. local pipeline fwd-bwd
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 8 GPU
# a. local pipeline fwd-bwd
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


data_fwd_mean = [
    [
        962.80, 
        959.73,
        955.76,
        946.43,
    ], # local pipeline
    [
        948.57,
        953.27,
        949.69,
        948.76,
    ], # CPU RPC
    [
        1160.06,
        1105.11,
        1103.12,
        1151.98,
    ], # CUDA RPC
]

data_fwd_stdv = [
    [
        1.26, 
        1.76, 
        2.08,
        2.25,
    ], # local pipeline
    [
        0.96,
        2.00,
        2.13,
        2.52,
    ], # CPU RPC
    [
        18.82,
        128.73,
        185.94,
        644.52,
    ], # CUDA RPC
]

data_comm_mean = [
    [
        0,
        0, 
        0,
        0,
    ],
    [
        580.67,
        636.46,
        949.69,
        1279.16,
    ],
    [
        36.34,
        24.44,
        16.17,
        138.47,
    ],
]

data_comm_stdv = [
    [
        0,
        0, 
        0,
        0,
    ],
    [
        4.62,
        5.17,
        24.47,
        17.80,
    ],
    [
        1.40,
        0.85,
        0.52,
        40.4,
    ],
]

data_bwd_mean = [
    [
        1748.39,
        1735.81,
        1729.72,
        1716.74,
    ],
    [
        4055.51,
        4252.61,
        4556.99,
        5577.68,
    ],
    [
        1775.24,
        1744.60,
        1741.34,
        1836.21,
    ],
]

data_bwd_stdv = [
    [
        3.67,
        4.63,
        5.79,
        4.84,
    ],
    [
        32.02,
        84.56,
        56.84,
        150.76
    ],
    [
        5.02,
        4.21,
        5.65,
        45.48,
    ],
]


# figure 2
# BS = 32 nGPU = 8

# 1 machine
# a. local pipeline
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd

# 2 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 4 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 8 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd

# figure 3
# BS = 128 nGPU = 8

# 1 machine
# a. local pipeline
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd

# 2 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 4 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 8 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd

# figure 4
# BS = 256 nGPU = 8

# 1 machine
# a. local pipeline
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd

# 2 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 4 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 8 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd
