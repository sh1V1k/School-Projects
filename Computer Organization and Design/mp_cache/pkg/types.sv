package cache_types;

typedef struct packed {
    logic [31:0] addr;
    logic [31:0] w_data;
    logic [3:0]  r_mask;
    logic [3:0]  w_mask;
} cpu_request_t;

typedef struct packed {
    logic [31:0] r_data;
    logic        response;
} cpu_response_t;

typedef struct packed {
    logic [31:0]  addr;
    logic         read;
    logic         write;
    logic [255:0] w_data;
} mem_request_t;

typedef struct packed {
    logic   [255:0] r_data;
    logic           response;
} mem_response_t;


endpackage