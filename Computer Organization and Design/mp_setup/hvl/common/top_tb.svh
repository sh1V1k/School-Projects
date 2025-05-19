    int timeout;
    initial begin
        timeout = 8;
    end

    always @(posedge clk) begin
        if (timeout == 0) begin
            $finish;
        end
        timeout <= timeout - 1;
    end

    logic   [2:0]   aluop;
    logic   [31:0]  a;
    logic   [31:0]  b;
    logic   [31:0]  f;

    alu dut(.*);

    initial begin
        a = 32'h800055AA;
        b = 32'h00000004;
        aluop = '0;
    end

    always @(posedge clk) begin
        aluop <= aluop + 3'd1;
    end
