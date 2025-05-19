.section .text
    .globl _start
_start:

    # Set x1 to a valid address within range (0xAAAAA000)
    lui   x1, 0xAAAAA      # Load upper 20 bits (0xAAAAA000)
    nop
    nop
    nop
    nop
    nop
    addi  x1, x1, 2000    # Add lower 12 bits (0xBBB)
    nop
    nop
    nop
    nop
    nop

    # Load Test Cases
    lw      x2, 0(x1)        # Load word (from 0xAAAAA000)
    nop
    nop
    nop
    nop
    nop

    lh      x3, 0(x1)        # Load half-word (lower 16 bits)
    nop
    nop
    nop
    nop
    nop

    lh      x4, 2(x1)        # Load half-word (upper 16 bits)
    nop
    nop
    nop
    nop
    nop

    lhu     x5, 0(x1)        # Load half-word unsigned
    nop
    nop
    nop
    nop
    nop

    lhu     x6, 2(x1)        # Load half-word unsigned
    nop
    nop
    nop
    nop
    nop

    lb      x7, 0(x1)        # Load byte (LSB)
    nop
    nop
    nop
    nop
    nop

    lb      x8, 1(x1)        # Load byte (Second byte)
    nop
    nop
    nop
    nop
    nop
    lb      x9, 2(x1)        # Load byte (Third byte)
    nop
    nop
    nop
    nop
    nop

    lb      x10, 3(x1)       # Load byte (MSB)
    nop
    nop
    nop
    nop
    nop

    lbu     x11, 0(x1)       # Load byte unsigned (LSB)
    nop
    nop
    nop
    nop
    nop

    lbu     x12, 1(x1)       # Load byte unsigned
    nop
    nop
    nop
    nop
    nop

    lbu     x13, 2(x1)       # Load byte unsigned
    nop
    nop
    nop
    nop
    nop

    lbu     x14, 3(x1)       # Load byte unsigned
    nop
    nop
    nop
    nop
    nop

    # Store Test Cases
    addi      x15, x15, 0x23  # Test value for stores
    nop
    nop
    nop
    nop
    nop

    sw      x15, 0(x1)       # Store word at 0xAAAAA000
    nop
    nop
    nop
    nop
    nop

    sh      x15, 4(x1)       # Store half-word at next half-word location
    nop
    nop
    nop
    nop
    nop

    sb      x15, 6(x1)       # Store byte at the next byte location
    nop
    nop
    nop
    nop
    nop

    # Load back stored values to verify correctness
    lw      x16, 0(x1)       # Load back stored word (should be 0xDEADBEEF)
    nop
    nop
    nop
    nop
    nop

    lh      x17, 4(x1)       # Load back stored half-word (should be 0xBEEF)
    nop
    nop
    nop
    nop
    nop

    lhu     x18, 4(x1)       # Load back stored half-word unsigned (should be 0xBEEF)
    nop
    nop
    nop
    nop
    nop

    lb      x19, 6(x1)       # Load back stored byte (should be 0xEF)
    nop
    nop
    nop
    nop
    nop

    lbu     x20, 6(x1)       # Load back stored byte unsigned (should be 0xEF)
    nop
    nop
    nop
    nop
    nop

    # End of test cases
    slti x0, x0, -256 # This is the magic instruction to end the simulation
    nop               # Preventing fetching illegal instructions
    nop
    nop
    nop
    nop