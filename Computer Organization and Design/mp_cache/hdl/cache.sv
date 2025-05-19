<<<<<<< HEAD
module cache 
import cache_types::*;
(
=======
module cache (
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc
    input   logic           clk,
    input   logic           rst,

    // cpu side signals, ufp -> upward facing port
    input   logic   [31:0]  ufp_addr,
    input   logic   [3:0]   ufp_rmask,
    input   logic   [3:0]   ufp_wmask,
    output  logic   [31:0]  ufp_rdata,
    input   logic   [31:0]  ufp_wdata,
    output  logic           ufp_resp,

    // memory side signals, dfp -> downward facing port
    output  logic   [31:0]  dfp_addr,
    output  logic           dfp_read,
    output  logic           dfp_write,
    input   logic   [255:0] dfp_rdata,
    output  logic   [255:0] dfp_wdata,
    input   logic           dfp_resp
);

<<<<<<< HEAD
localparam WAY = 4;
localparam S_INDEX = 4;
localparam NUM_SETS = 2**S_INDEX;
localparam TAGMSB = 31;
localparam TAGLSB = 9;

enum integer unsigned {
    idle, cmp_tag, alloc, wb
} state, state_next;

cpu_request_t cpu_request;
cpu_response_t cpu_response;
mem_request_t mem_request;
mem_response_t mem_response;

always_ff @ (posedge clk) begin
    if(rst) begin
        cpu_request <= '0;
        mem_response <= '0;
    end else begin
        cpu_request.addr <= ufp_addr;
        cpu_request.r_mask <= ufp_rmask;
        cpu_request.w_mask <= ufp_wmask;
        cpu_request.w_data <= ufp_wdata;

        mem_response.r_data <= dfp_rdata;
        mem_response.response <= dfp_resp;
    end
end

always_ff @ (posedge clk) begin
    if(rst) begin
        dfp_addr <= '0;
        dfp_read <= '0;
        dfp_write <= '0;
        dfp_wdata <= '0;
    end else begin
        //ufp_rdata <= cpu_response.r_data;
        //ufp_resp <= cpu_response.response;

        dfp_addr <= mem_request.addr;
        dfp_read <= mem_request.read;
        dfp_write <= mem_request.write;
        dfp_wdata <= mem_request.w_data;
    end
end

logic hit;

assign ufp_resp = hit;
assign ufp_rdata = cpu_response.r_data;

// always_comb begin
//     ufp_rdata = cpu_response.r_data;
//     ufp_resp = cpu_response.response;

//     dfp_addr = mem_request.addr;
//     dfp_read = mem_request.read;
//     dfp_write = mem_request.write;
//     dfp_wdata = mem_request.w_data;
// end

logic data_csb0 [WAY];
logic data_web0 [WAY];
logic [31:0] data_wmask0 [WAY];
logic [3:0] data_addr0 [WAY];
logic [255:0] data_din0 [WAY];
logic [255:0] data_dout0 [WAY];

logic tag_csb0 [WAY];
logic tag_web0 [WAY];
logic [3:0] tag_addr0 [WAY];
logic [22:0] tag_din0 [WAY];
logic [22:0] tag_dout0 [WAY];

logic valid_csb0 [WAY];
logic valid_web0 [WAY];
logic [3:0] valid_addr0 [WAY];
logic valid_din0 [WAY];
logic valid_dout0 [WAY];

logic dirty_csb0 [WAY];
logic dirty_web0 [WAY];
logic [3:0] dirty_addr0 [WAY];
logic dirty_din0 [WAY];
logic dirty_dout0 [WAY];

logic lru_csb0;
logic lru_web0;
logic [3:0] lru_addr0;
logic [2:0] lru_din0;
logic [2:0] lru_dout0;

logic [1:0] lru_decode;
    generate for (genvar i = 0; i < WAY; i++) begin : arrays //4 is for the four way, the 4 inside these if for the 16 sets
        mp_cache_data_array data_array (
            .clk0       (clk),
            .csb0       (data_csb0[i]),
            .web0       (data_web0[i]),
            .wmask0     (data_wmask0[i]),
            .addr0      (data_addr0[i]),
            .din0       (data_din0[i]),
            .dout0      (data_dout0[i])
        );
        mp_cache_tag_array tag_array (
            .clk0       (clk),
            .csb0       (tag_csb0[i]),
            .web0       (tag_web0[i]),
            .addr0      (tag_addr0[i]),
            .din0       (tag_din0[i]),
            .dout0      (tag_dout0[i])
        );
        sp_ff_array valid_array (
            .clk0       (clk),
            .rst0       (rst),
            .csb0       (valid_csb0[i]),
            .web0       (valid_web0[i]),
            .addr0      (valid_addr0[i]),
            .din0       (valid_din0[i]),
            .dout0      (valid_dout0[i])
        );
        sp_ff_array dirty_array (
            .clk0       (clk),
            .rst0       (rst),
            .csb0       (dirty_csb0[i]),
            .web0       (dirty_web0[i]),
            .addr0      (dirty_addr0[i]),
            .din0       (dirty_din0[i]),
            .dout0      (dirty_dout0[i])
        );
    end 
endgenerate

    // have 3 bits per set and have 16*4 sets which is why we change the width to 3
    sp_ff_array #(
        .WIDTH      (3)
    ) lru_array (
        .clk0       (clk),
        .rst0       (rst),
        .csb0       (lru_csb0),
        .web0       (lru_web0),
        .addr0      (lru_addr0),
        .din0       (lru_din0),
        .dout0      (lru_dout0)
    );

always_ff @ (posedge clk) begin
    if(rst)
        state <= idle;
    else
        state <= state_next;
end

always_comb begin
unique  casez (lru_dout0)
                    3'b00?: lru_decode = 2'd0;
                    3'b01?: lru_decode = 2'd1;
                    3'b1?0: lru_decode = 2'd2;
                    3'b1?1: lru_decode = 2'd3;
                    default: lru_decode = 'x;
        endcase
end

always_comb begin 
    state_next = state;
    hit = '0;

    for(integer i = 0; i < WAY; i++) begin
    dirty_csb0[i] = '0;
    dirty_web0[i] = '1;
    dirty_addr0[i] ='0;
    dirty_din0[i] = '0; 

    tag_csb0[i] = '0;
    tag_web0[i] = '1;
    tag_addr0[i] ='0;
    tag_din0[i] = '0;

    valid_csb0[i] = '0;
    valid_web0[i] = '1;
    valid_addr0[i] ='0;
    valid_din0[i] = '0;

    data_csb0[i] = '0;
    data_web0[i] = '1;
    data_addr0[i] = '0;
    data_wmask0[i] = '0;
    data_din0[i] = '0;

    lru_csb0 = '0;
    lru_web0 = '1;
    lru_addr0 ='0;
    lru_din0 = '0;
    end

    cpu_response.r_data = '0;
    cpu_response.response = '0;
    mem_request.addr = '0;
    mem_request.write = '0;
    mem_request.read = '0;
    mem_request.w_data = '0;


    unique case (state)


        idle: begin
            if(ufp_rmask != 0 || ufp_wmask != 0)
                state_next = cmp_tag;

                //stuff to access next cycle
            for(integer i = 0; i < WAY; i++) begin
                tag_addr0[i] = ufp_addr[8:5];
                data_addr0[i] = ufp_addr[8:5];
                valid_addr0[i] = ufp_addr[8:5];
                lru_addr0 = ufp_addr[8:5];
                dirty_addr0[i] = ufp_addr[8:5];
            end
        end

        cmp_tag: begin
            for(integer unsigned i = 0; i < WAY; i++)  begin
                tag_addr0[i] = cpu_request.addr[8:5];

                data_addr0[i] = cpu_request.addr[8:5];

                valid_addr0[i] = cpu_request.addr[8:5];

                lru_addr0 = ufp_addr[8:5];
                //cache hit
                if(valid_dout0[i] && tag_dout0[i] == cpu_request.addr[TAGMSB:TAGLSB]) begin
                    cpu_response.response = '1;
                    state_next = idle;
                    hit = '1;

                    //if(i == {30'b0, lru_decode}) begin
                            lru_csb0 = '0;
                            lru_web0 = '0;
                            lru_addr0 = cpu_request.addr[8:5];
                            lru_din0 = lru_dout0;

                            unique case (i) 
                                32'd0 : begin lru_din0[2] = 1'b1; lru_din0[1] = 1'b1; end
                                32'd1 : begin lru_din0[2] = 1'b1; lru_din0[1] = 1'b0; end
                                32'd2 : begin lru_din0[2] = 1'b0; lru_din0[0] = 1'b1; end
                                32'd3 : begin lru_din0[2] = 1'b0; lru_din0[0] = 1'b0; end 
                                default : lru_din0 = 'x;
                            endcase
                            //lru_din0 = lru_decode < 2 ? {lru_dout0[2],~lru_dout0[1],~lru_dout0[0]} : {~lru_dout0[2],lru_dout0[1],~lru_dout0[0]};
                    //end

                    //write -- mark set dirty, data
                    if(cpu_request.w_mask > 0) begin
                        dirty_web0[i] = '0;
                        dirty_addr0[i] = cpu_request.addr[8:5];
                        dirty_din0[i] = '1; 

                        data_web0[i] = '0;
                        data_addr0[i] = cpu_request.addr[8:5];
                        data_wmask0[i][4*cpu_request.addr[4:2] +: 4] = cpu_request.w_mask;
                        data_din0[i][32*cpu_request.addr[4:2] +: 32] = cpu_request.w_data; 

                    end else begin //read, set valid and tag and update lru, dont have to care about rmask since we can just send all 4 bytes over
                        cpu_response.r_data = data_dout0[i][32*cpu_request.addr[4:2] +: 32]; //should contain correct data since we requested last clock cycle
                    end

                    break; //if we hit we no longer need to continue loop
                end
            end

            //cache miss
            if(~hit) begin
                lru_addr0 = cpu_request.addr[8:5];

                if(dirty_dout0[lru_decode] && valid_dout0[lru_decode]) begin
                    state_next = wb;

                    mem_request.addr = {tag_dout0[lru_decode], cpu_request.addr[8:5], {5{1'b0}}};
                    mem_request.write = '1;
                    mem_request.read = '0;
                    mem_request.w_data = data_dout0[lru_decode];
                end else begin
                    state_next = alloc;

                    mem_request.addr = {cpu_request.addr[TAGMSB:TAGLSB], cpu_request.addr[8:5],{5{1'b0}}};
                    mem_request.write = '0;
                    mem_request.read = '1;
                    mem_request.w_data = '0;
                end

            end

        end

        alloc: begin
            lru_csb0 = '0;
            lru_web0 = '1;
            lru_addr0 = cpu_request.addr[8:5];

            mem_request.addr = {cpu_request.addr[TAGMSB:TAGLSB], cpu_request.addr[8:5],{5{1'b0}}};
            mem_request.write = '0;
            mem_request.read = '1;
            mem_request.w_data = '0;

            if(dfp_resp) begin //update valid, tag, data(all of it)
                valid_csb0[lru_decode] = '0;
                valid_web0[lru_decode] = '0;
                valid_addr0[lru_decode] = cpu_request.addr[8:5];
                valid_din0[lru_decode] = '1;

                dirty_csb0[lru_decode] = '0;
                dirty_web0[lru_decode] = '0;
                dirty_addr0[lru_decode] = cpu_request.addr[8:5];
                dirty_din0[lru_decode] = '0;

                data_csb0[lru_decode] = '0;
                data_web0[lru_decode] = '0;
                data_addr0[lru_decode] = cpu_request.addr[8:5];
                data_wmask0[lru_decode] = '1; //update entire cache line
                data_din0[lru_decode] = dfp_rdata; 

                tag_csb0[lru_decode] = '0;
                tag_web0[lru_decode] = '0;
                tag_addr0[lru_decode] = cpu_request.addr[8:5];
                tag_din0[lru_decode] = cpu_request.addr[TAGMSB:TAGLSB];

                state_next = idle;

                mem_request.addr = '0;;
                mem_request.write = '0;
                mem_request.read = '0;
                mem_request.w_data = '0;
            end

        end

        wb: begin //move to alloc

        lru_csb0 = '0;
        lru_web0 = '1;
        lru_addr0 = cpu_request.addr[8:5];

        tag_addr0[lru_decode] = cpu_request.addr[8:5];

        data_addr0[lru_decode] = cpu_request.addr[8:5];

            mem_request.addr = {tag_dout0[lru_decode], cpu_request.addr[8:5],{5{1'b0}}};
            mem_request.write = '1;
            mem_request.read = '0;
            mem_request.w_data = data_dout0[lru_decode];

            if(dfp_resp)begin
                state_next = alloc;

                mem_request.addr = {cpu_request.addr[TAGMSB:TAGLSB], cpu_request.addr[8:5],{5{1'b0}}};
                mem_request.write = '0;
                mem_request.read = '1;
                mem_request.w_data = '0;
            end
        end
    
    default: begin
        state_next = idle;
    end

    endcase
end

=======
    generate for (genvar i = 0; i < 4; i++) begin : arrays
        mp_cache_data_array data_array (
            .clk0       (),
            .csb0       (),
            .web0       (),
            .wmask0     (),
            .addr0      (),
            .din0       (),
            .dout0      ()
        );
        mp_cache_tag_array tag_array (
            .clk0       (),
            .csb0       (),
            .web0       (),
            .addr0      (),
            .din0       (),
            .dout0      ()
        );
        sp_ff_array valid_array (
            .clk0       (),
            .rst0       (),
            .csb0       (),
            .web0       (),
            .addr0      (),
            .din0       (),
            .dout0      ()
        );
    end endgenerate

    sp_ff_array #(
        .WIDTH      (3)
    ) lru_array (
        .clk0       (),
        .rst0       (),
        .csb0       (),
        .web0       (),
        .addr0      (),
        .din0       (),
        .dout0      ()
    );

>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc
endmodule
