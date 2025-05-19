<<<<<<< HEAD
# .section .text
# .globl _start
# _start:
#     # ADD - Add
#     add x5, x1, x2
#     nop
#     nop
#     nop
#     nop
#     nop

#     # SUB - Subtract
#     sub x6, x3, x4
#     nop
#     nop
#     nop
#     nop
#     nop

#     # XOR - Bitwise XOR
#     xor x10, x4, x6
#     nop
#     nop
#     nop
#     nop
#     nop

#     # OR - Bitwise OR
#     or x11, x5, x7
#     nop
#     nop
#     nop
#     nop
#     nop

#     # AND - Bitwise AND
#     and x12, x8, x9
#     nop
#     nop
#     nop
#     nop
#     nop

#     # SLL - Shift Left Logical
#     sll x13, x10, x1
#     nop
#     nop
#     nop
#     nop
#     nop

#     # SRL - Shift Right Logical
#     srl x14, x11, x2
#     nop
#     nop
#     nop
#     nop
#     nop

#     # SRA - Shift Right Arithmetic
#     sra x15, x12, x3
#     nop
#     nop
#     nop
#     nop
#     nop

#     # LUI - Load Upper Immediate
#     lui x16, 0x12345
#     nop
#     nop
#     nop
#     nop
#     nop

#     # AUIPC - Add Upper Immediate to PC
#     auipc x17, 0x6789
#     nop
#     nop
#     nop
#     nop
#     nop

#     # ADDI - Add Immediate
#     addi x18, x1, 100
#     nop
#     nop
#     nop
#     nop
#     nop

#     # SLTI - Set Less Than Immediate
#     slti x19, x2, 50
#     nop
#     nop
#     nop
#     nop
#     nop

#     # SLTIU - Set Less Than Immediate Unsigned
#     sltiu x20, x3, 75
#     nop
#     nop
#     nop
#     nop
#     nop

#     # XORI - XOR Immediate
#     xori x21, x4, 0x55
#     nop
#     nop
#     nop
#     nop
#     nop

#     # ORI - OR Immediate
#     ori x22, x5, 0xAA
#     nop
#     nop
#     nop
#     nop
#     nop

#     # ANDI - AND Immediate
#     andi x23, x6, 0xFF
#     nop
#     nop
#     nop
#     nop
#     nop

#     # SLLI - Shift Left Logical Immediate
#     slli x24, x7, 3
#     nop
#     nop
#     nop
#     nop
#     nop

#     # SRLI - Shift Right Logical Immediate
#     srli x25, x8, 2
#     nop
#     nop
#     nop
#     nop
#     nop

#     # SRAI - Shift Right Arithmetic Immediate
#     srai x26, x9, 1
#     nop
#     nop
#     nop
#     nop
#     nop

#     slti x0, x0, -256 # this is the magic instruction to end the simulation
#     nop               # preventing fetching illegal instructions
#     nop
#     nop
#     nop
#     nop

        .text
    .globl _start
_start:
    # ADDI – Immediate Addition Cases
    # 0 + 0 = 0
    addi    x1, x0, 0           
=======
.section .text
.globl _start
_start:
    addi x1, x0, 4
    nop             # nops in between to prevent hazard
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc
    nop
    nop
    nop
    nop
<<<<<<< HEAD
    nop

    # 0 + 100 = 100 (normal positive)
    addi    x2, x0, 100         
    nop
    nop
    nop
    nop
    nop

    # Maximum positive immediate (2047)
    addi    x3, x0, 2047        
    nop
    nop
    nop
    nop
    nop

    # 0 + (-100) = -100 (normal negative)
    addi    x4, x0, -100        
    nop
    nop
    nop
    nop
    nop

    # Minimum negative immediate (-2048)
    addi    x5, x0, -2048       
    nop
    nop
    nop
    nop
    nop

    # Immediate -1 (all ones when sign‑extended)
    addi    x6, x0, -1          
    nop
    nop
    nop
    nop
    nop

    # ANDI, ORI, XORI – Bitwise Immediates
    # AND with 0xF (masking lower 4 bits)
    andi    x7, x1, 0xF         
    nop
    nop
    nop
    nop
    nop

    # AND with all ones (should pass x2 through)
    andi    x8, x2, -1          
    nop
    nop
    nop
    nop
    nop

    # AND with 0xAA pattern
    andi    x9, x3, 0xAA        
    nop
    nop
    nop
    nop
    nop

    # OR with 0xF (setting lower 4 bits)
    ori     x10, x1, 0xF        
    nop
    nop
    nop
    nop
    nop

    # OR with all ones (result is all ones)
    ori     x11, x2, -1         
    nop
    nop
    nop
    nop
    nop

    # OR with 0x55 pattern
    ori     x12, x3, 0x55        
    nop
    nop
    nop
    nop
    nop

    # XOR with 0xF (flip lower 4 bits)
    xori    x13, x1, 0xF        
    nop
    nop
    nop
    nop
    nop

    # XOR with -1 (bitwise complement of x2)
    xori    x14, x2, -1         
    nop
    nop
    nop
    nop
    nop

    # XOR with 0xAA pattern
    xori    x15, x3, 0xAA       
    nop
    nop
    nop
    nop
    nop

    # Shift Immediate Instructions
    # Shift left immediate by 0 (no change)
    slli    x16, x1, 0          
    nop
    nop
    nop
    nop
    nop

    # Shift left immediate by 1
    slli    x17, x2, 1          
    nop
    nop
    nop
    nop
    nop

    # Shift left immediate by 15 (mid-range)
    slli    x18, x2, 15         
    nop
    nop
    nop
    nop
    nop

    # Shift left immediate by 31 (edge case)
    slli    x19, x2, 31         
    nop
    nop
    nop
    nop
    nop

    # Logical shift right immediate by 0 (no change)
    srli    x20, x3, 0          
    nop
    nop
    nop
    nop
    nop

    # Logical shift right immediate by 1
    srli    x21, x3, 1          
    nop
    nop
    nop
    nop
    nop

    # Logical shift right immediate by 15
    srli    x22, x3, 15         
    nop
    nop
    nop
    nop
    nop

    # Logical shift right immediate by 31 (edge)
    srli    x23, x3, 31         
    nop
    nop
    nop
    nop
    nop

    # Arithmetic shift right immediate by 0 (no change)
    srai    x24, x4, 0          
    nop
    nop
    nop
    nop
    nop

    # Arithmetic shift right immediate by 1
    srai    x25, x4, 1          
    nop
    nop
    nop
    nop
    nop

    # Arithmetic shift right immediate by 15
    srai    x26, x4, 15         
    nop
    nop
    nop
    nop
    nop

    # Arithmetic shift right immediate by 31 (tests sign extension)
    srai    x27, x4, 31         
    nop
    nop
    nop
    nop
    nop

    # Register Arithmetic: ADD & SUB
    # ADD: x2 + x3 (normal operation)
    add     x28, x2, x3         
    nop
    nop
    nop
    nop
    nop

    # ADD: adding two negative numbers (x4 + x5)
    add     x29, x4, x5         
    nop
    nop
    nop
    nop
    nop

    # SUB: x2 - x3 (could underflow)
    sub     x30, x2, x3         
    nop
    nop
    nop
    nop
    nop

    # SUB: subtracting negatives (x4 - x5)
    sub     x31, x4, x5         
    nop
    nop
    nop
    nop
    nop

    # Bitwise Register Operations
    # Bitwise AND of x2 and x3
    and     x1, x2, x3          
    nop
    nop
    nop
    nop
    nop

    # Bitwise OR of x2 and x3
    or      x2, x2, x3          
    nop
    nop
    nop
    nop
    nop

    # Bitwise XOR of x2 and x3
    xor     x3, x2, x3          
    nop
    nop
    nop
    nop
    nop

    # Register Shift Instructions (shift amount from register; only lower 5 bits are used in RV32)
    # Set shift amount = 2
    addi    x4, x0, 2           
    nop
    nop
    nop
    nop
    nop

    # Shift left register x2 by value in x4
    sll     x5, x2, x4          
    nop
    nop
    nop
    nop
    nop

    # Logical shift right register x3 by x4
    srl     x6, x3, x4          
    nop
    nop
    nop
    nop
    nop

    # Arithmetic shift right; shifting x4 by itself (for variety)
    sra     x7, x4, x4          
    nop
    nop
    nop
    nop
    nop

    # Test with shift amounts larger than 31 (only lower 5 bits count)
    addi    x8, x0, 33          # 33 mod 32 = 1
    nop
    nop
    nop
    nop
    nop

    # Shift left register x2 by x8 (should shift by 1)
    sll     x9, x2, x8          
    nop
    nop
    nop
    nop
    nop

    # Logical shift right register x3 by x8 (should shift by 1)
    srl     x10, x3, x8         
    nop
    nop
    nop
    nop
    nop

    # Arithmetic shift right register x4 by x8 (should shift by 1)
    sra     x11, x4, x8         
    nop
    nop
    nop
    nop
    nop

    # Set Less Than Instructions (SLT & SLTU)
    # Set x12 = 10
    addi    x12, x0, 10         
    nop
    nop
    nop
    nop
    nop

    # Set x13 = 20
    addi    x13, x0, 20         
    nop
    nop
    nop
    nop
    nop

    # SLT: 10 < 20 should yield 1
    slt     x14, x12, x13       
    nop
    nop
    nop
    nop
    nop

    # SLT: 20 < 10 should yield 0
    slt     x15, x13, x12       
    nop
    nop
    nop
    nop
    nop

    # Set x16 = -1 (0xFFFFFFFF)
    addi    x16, x0, -1         
    nop
    nop
    nop
    nop
    nop

    # SLTU: 0xFFFFFFFF < 10 should yield 0
    sltu    x17, x16, x12       
    nop
    nop
    nop
    nop
    nop

    # SLTU: 10 < 0xFFFFFFFF should yield 1
    sltu    x18, x12, x16       
    nop
    nop
    nop
    nop
    nop

    # Set x19 = 0
    addi    x19, x0, 0          
    nop
    nop
    nop
    nop
    nop

    # Set x20 = 1
    addi    x20, x0, 1          
    nop
    nop
    nop
    nop
    nop

    # SLT: 0 < 1 should yield 1
    slt     x21, x19, x20       
    nop
    nop
    nop
    nop
    nop

    # SLT: 1 < 0 should yield 0
    slt     x22, x20, x19       
    nop
    nop
    nop
    nop
    nop

    # LUI – Load Upper Immediate
    # LUI: loads 0x12345 << 12 into x23
    lui     x23, 0x12345        
    nop
    nop
    nop
    nop
    nop

    # LUI: maximum 20-bit immediate (edge)
    lui     x24, 0xFFFFF        
    nop
    nop
    nop
    nop
    nop

    # LUI: minimal nonzero upper immediate
    lui     x25, 0x00001        
    nop
    nop
    nop
    nop
    nop

    # AUIPC – Add Upper Immediate to PC
    # AUIPC: PC + (0x00010 << 12)
    auipc   x26, 0x00010        
    nop
    nop
    nop
    nop
    nop

    # AUIPC: large immediate offset (edge)
    auipc   x27, 0x7FFFF        
    nop
    nop
    nop
    nop
    nop

    # AUIPC: minimal immediate
    auipc   x28, 0x00001        
    nop
    nop
    nop
    nop
    nop

    # Additional Normal Operations
    addi    x29, x0, 123        
    nop
    nop
    nop
    nop
    nop

    add     x30, x29, x2        
    nop
    nop
    nop
    nop
    nop

    and     x31, x29, x2        
    nop
    nop
    nop
    nop
    nop

    or      x1, x29, x2         
    nop
    nop
    nop
    nop
    nop

    xor     x2, x29, x2         
    nop
    nop
    nop
    nop
    nop


    addi    x1, x0, 0         # x1 = 0
    nop
    nop
    nop
    nop
    nop
    slti    x2, x1, 0         # is 0 < 0? (expected: 0)
    nop
    nop
    nop
    nop
    nop

    # Test 2: -1 < 0 → true
    addi    x3, x0, -1        # x3 = -1
    nop
    nop
    nop
    nop
    nop
    slti    x4, x3, 0         # is -1 < 0? (expected: 1)
    nop
    nop
    nop
    nop
    nop

    # Test 3: 5 < 10 → true
    addi    x5, x0, 5         # x5 = 5
    nop
    nop
    nop
    nop
    nop
    slti    x6, x5, 10        # is 5 < 10? (expected: 1)
    nop
    nop
    nop
    nop
    nop

    # Test 4: 10 < 5 → false
    addi    x7, x0, 10        # x7 = 10
    nop
    nop
    nop
    nop
    nop
    slti    x8, x7, 5         # is 10 < 5? (expected: 0)
    nop
    nop
    nop
    nop
    nop

    # Test 5: -10 < -5 → true
    addi    x9, x0, -10       # x9 = -10
    nop
    nop
    nop
    nop
    nop
    slti    x10, x9, -5       # is -10 < -5? (expected: 1)
    nop
    nop
    nop
    nop
    nop

    # Test 6: -10 < -10 → false
    addi    x11, x0, -10      # x11 = -10
    nop
    nop
    nop
    nop
    nop
    slti    x12, x11, -10     # is -10 < -10? (expected: 0)
    nop
    nop
    nop
    nop
    nop

    # Test 7: 2047 < 2047 → false
    addi    x13, x0, 2047     # x13 = 2047 (max positive immediate)
    nop
    nop
    nop
    nop
    nop
    slti    x14, x13, 2047    # is 2047 < 2047? (expected: 0)
    nop
    nop
    nop
    nop
    nop

    # Test 8: -2048 < -2048 → false
    addi    x15, x0, -2048    # x15 = -2048 (min negative immediate)
    nop
    nop
    nop
    nop
    nop
    slti    x16, x15, -2048   # is -2048 < -2048? (expected: 0)
    nop
    nop
    nop
    nop
    nop

    # Test 9: -2048 < 2047 → true
    addi    x17, x0, -2048    # x17 = -2048
    nop
    nop
    nop
    nop
    nop
    slti    x18, x17, 2047    # is -2048 < 2047? (expected: 1)
    nop
    nop
    nop
    nop
    nop

    # Test 10: 2047 < -2048 → false
    addi    x19, x0, 2047     # x19 = 2047
    nop
    nop
    nop
    nop
    nop
    slti    x20, x19, -2048   # is 2047 < -2048? (expected: 0)
    nop
    nop
    nop
    nop
    nop

    # Test 11: -1 < -1 → false
    addi    x21, x0, -1       # x21 = -1
    nop
    nop
    nop
    nop
    nop
    slti    x22, x21, -1      # is -1 < -1? (expected: 0)
    nop
    nop
    nop
    nop
    nop

    # Test 12: 100 < -1 → false
    addi    x23, x0, 100      # x23 = 100
    nop
    nop
    nop
    nop
    nop
    slti    x24, x23, -1      # is 100 < -1? (expected: 0)
    nop
    nop
    nop
    nop
    nop

    # Test 13: -100 < 100 → true
    addi    x25, x0, -100     # x25 = -100
    nop
    nop
    nop
    nop
    nop
    slti    x26, x25, 100     # is -100 < 100? (expected: 1)
    nop
    nop
    nop
    nop
    nop

    lui     x10, 0xaaaab
    nop
    nop
    nop
    nop
    nop

    lui     x7, 0x33333
    nop
    nop
    nop
    nop
    nop

    lui     x7, 0x59237
    nop
    nop
    nop
    nop
    nop

    lui     x7, 0x88742
    nop
    nop
    nop
    nop
    nop

    auipc  x22, 333
    nop
    nop
    nop
    nop
    nop

    auipc  x21, 4999
    nop
    nop
    nop
    nop
    nop

    auipc  x30, 9655
=======
    addi x3, x1, 8
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc
    nop
    nop
    nop
    nop
    nop

<<<<<<< HEAD
    # End of test cases
=======
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc
    slti x0, x0, -256 # this is the magic instruction to end the simulation
    nop               # preventing fetching illegal instructions
    nop
    nop
    nop
    nop
