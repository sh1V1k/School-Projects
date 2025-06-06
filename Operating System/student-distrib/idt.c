#include "idt.h"
#include "keyboard.h"
#include "rtc.h"
#include "syscall.h"

// list of all exceptions - used in handler
char* exceptions_list[] = {
    "divide_error",
    "debug",
    "nmi",
    "breakpoint",
    "overflow",
    "bounds", 
    "opcode", 
    "coprocessor",
    "double_fault",
    "segment_overun",
    "invalid_tss",
    "seg_not_present",
    "stack_fault",
    "general_protection_fault",
    "page_fault",
    "reserved",
    "math_fault",
    "alignment_check",
    "machine_check",
    "simd_math_fault",
    "virtualization_fault"
}; //list of possible exceptions we handle

/* void idt_init()
 *
 * initalizes IDT
 * Inputs: None
 * Outputs: Void
 * Side Effects: IDT activated
 */
void idt_init(){
    int i;
    for(i = 0; i < NUM_VEC; i++){
      if ( i < 32){  idt[i].present = 1; }
      else {
          idt[i].present = 0;
      }
        //setting the rest of the bits
        idt[i].seg_selector = KERNEL_CS;
        idt[i].dpl = 0; //have to change for syscalls
        idt[i].reserved0 = 0; //! figure out where this is from 
        idt[i].reserved1 = 1;
        idt[i].reserved2 = 1;
        idt[i].reserved3 = 1;     
        idt[i].reserved4 = 0;
        idt[i].size = 1;
      
    }
    // handlers[0] = divide_error;
    // handlers[1] = debug;
    // handlers[2] = nmi;
    // handlers[3] = breakpoint;
    // handlers[4] = overflow;
    // handlers[5] = bounds;
    // handlers[6] = opcode;
    // handlers[7] = coprocessor;
    // handlers[8] = 
    // handlers[9]
    // handlers[10]
    // handlers[11]
    // handlers[12]
    // handlers[13]
    // handlers[14]
    // handlers[16]

    SET_IDT_ENTRY(idt[0], divide_error);
    SET_IDT_ENTRY(idt[1], debug);
    SET_IDT_ENTRY(idt[2], nmi);
    SET_IDT_ENTRY(idt[3], breakpoint);
    SET_IDT_ENTRY(idt[4], overflow);
    SET_IDT_ENTRY(idt[5], bounds);
    SET_IDT_ENTRY(idt[6], opcode);
    SET_IDT_ENTRY(idt[7], coprocessor);
    SET_IDT_ENTRY(idt[8], double_fault);
    SET_IDT_ENTRY(idt[9], segment_overrun);
    SET_IDT_ENTRY(idt[10], invalid_tss);
    SET_IDT_ENTRY(idt[11], seg_not_present);
    SET_IDT_ENTRY(idt[12], stack_fault);
    SET_IDT_ENTRY(idt[13], general_protection_fault);
    SET_IDT_ENTRY(idt[14], page_fault);
    SET_IDT_ENTRY(idt[16], math_fault);
    SET_IDT_ENTRY(idt[17], alignment_check);
    SET_IDT_ENTRY(idt[18], machine_check);
    SET_IDT_ENTRY(idt[19], simd_math_fault);
    SET_IDT_ENTRY(idt[20], virtualization_fault); 
    
    //set correct idt values for syscall 
    //0x80 is vector for syscall
    SET_IDT_ENTRY(idt[SYSCALL_VEC], syscall);
    idt[SYSCALL_VEC].dpl = 3;

    SET_IDT_ENTRY(idt[KEYBOARD_VEC], KEYBOARD_INT_MACRO); //was keypress
    idt[KEYBOARD_VEC].reserved3 = 0;
    
    SET_IDT_ENTRY(idt[RTC_VEC], RTC_INT_MACRO); //setup all exceptions, was rtc_interrupt()
    idt[RTC_VEC].reserved3 = 0;

    SET_IDT_ENTRY(idt[PIT_VEC], PIT_INT_MACRO); //setup all exceptions, was rtc_interrupt()
    idt[PIT_VEC].reserved3 = 0;
    

    lidt(idt_desc_ptr);
}
/* void handler(uint32_t id, uint32_t fl, struct x86_regs regs, uint32_t err)
 *
 * Handles all interrupts
 * Inputs: id - vector number; fl - flags; regs - registers; err - error code
 * Outputs: Void
 * Side Effects: Prints text to blue screen
 */
void handler(uint32_t id, uint32_t fl, struct x86_regs regs, uint32_t err){
    // if(id == 0x80){
    //     //printf("int 0x80 executed\n");
    //     //int32_t execute(const uint8_t* command);
    // }
    // else if(id == 0x80){
    //     printf("Exception is: syscall \t id: 0x80");
    //     while(1) {}
    // }
    if (id < 20){
        printf("Exception is: %s \t id: %d\n", exceptions_list[id], id);
        //printf("welcome to the hannah montana blue sceen of death,\n to support multiculturalism and not discriminate \nagainst the blue people from avatar\nwe bring you this death screen\n");
        printf("ERROR: %d\n", err);  
        // uint32_t ebp;
        // uint32_t esp;
        // uint32_t eip;
        // uint32_t eflags;
        // asm("\t movl %%ebp, %0" : "=r"(ebp));

        // Hannah Montana ASCII Art

        /*printf("....................................................................................................\n");
    printf("....................................................................................................\n");
    printf("...............................................--------.............................................\n");
    printf(".........................................:=++++*%%**%%+:=-..........................................\n");
    printf(".......................................:--@%%@#:+*--**-:*+--........................................\n");
    printf("....................................-==*+:*++*=..=**++*==***=.......................................\n");
    printf("..................................:**+#=...-+#@*: ..-+#@+*=:=*=.....................................\n");
    printf(".................................=%:+.:**....=@=*+.*-.:-#@@*.:=%....................................\n");
    printf(".................................+%.=..*#-%- .:%=+%-**.:%++#.-=@....................................\n");
    printf("................................@*-@#%:.=%:....@@*:%==%:.**=@==@....................................\n");
    printf("................................@*-%.:%==@@--: .#*.%==%:.#*=@==@....................................\n");
    printf("................................#*=#%%:..:%@@@+:.=#@@*:%:.=+.%*:#+:.................................\n");
    printf(".................................+%.-+=:.:====@@#-..#*:%:--..%*. #+.................................\n");
    printf("..................................-%@%%=.:#%%%:::+@@=+@@:**-%+=%.#+.................................\n");
    printf("............................ ...  .. ... ...  ...-+@@*:%:**..==@.*+.................................\n");
    printf(".........................-@-....  .......-@-  ...-+@@*:#=*@#.#*=#@+.................................\n");
    printf("..................... =%:...-%=..:-@*..+#*:......-+@@*.=**##.%*:@@+.................................\n");
    printf("......................:-*%@+-.*@#%@@*:#%####%@...-+@@*:%:#@#.%*:@#-.................................\n");
    printf("..........:*=+%@@@@@@@@@@@@@@*+*+==@@#*%****+=...-+@@%*+*+#@**+*=*-:+%@@@#+.:+......................\n");
    printf("..........-@@@@@@@@@@@@@@#:+#@%%@@#:*@@+=@@=.....-%@@@%-%=.=@==@ #+-@@@@@*:%*:%#... .................\n");
    printf("............*@@@@@@@%. :%#....*@....*@@=. .  ..:#@@@@@%-%@*..%@@@=.-@@@@:+@:=@:-@-..................\n");
    printf("................... . ...-%-.....   +%#*:....:@@@@@@@@%-%@@#.%@%:#+..+%%@@%@@@@@@@@@=...............\n");
    printf("..................   .... :%#     ..*#*#@#-#@@@@@@%%%@%-%@%+@#+=#@*:....*-.@@@@@@@@@@@%+:...........\n");
    printf(".........................:+**=....:@@@@*=#@*++##@#-:.*%-%@@#.==@.#%*=:. .. @@#@@#####*=+*=-.........\n");
    printf(".....................  :**-:=%-..:=@@@=*#%-+%*..=#=:.+@#=#@@#%*=#@+=@-. ...@*.@#.    .=:-@+.........\n");
    printf("..................... +%-...=%-..:=@@%-.=@@-+%...:=:.*%-*:.+@@*:@@+=@@*  ..@#..-@-.. ...-@+.........\n");
    printf(".....................=**:...=%-..:=@#-%@%+#++#=. ....+.=@%*+@%+:@@@%.+*=....=@:..=+.....-@+.........\n");
    printf("...................=#@*.....=%-..:=@@#@#=+*+**+*-....=*++*+*@==#=*++%+++*-..=@@#........-@+.........\n");
    printf("...................*@%*.....=%-..:=@@@=*%#-#*:.@@+.  .-#::+*:#*-@++#-#@%-...=@@#........-@+.........\n");
    printf("..................@@@.*@:=@--....:=@@%-#@@@-+@@-#@@@@*....+#...:@+...#@@:...=@@#........-@+.........\n");
    printf("................=@@@#=#@@#+@%....:=@#-%@%*@-+@@+#@%*@@@@@@*=@@*-@@+::*+.#=..=@@#........=@+.........\n");
    printf("..............-**=%@@@*#@@@@%....:=@%=#@#-@+*@@%@%=@%%@%@@#+%++%-#@@%=..-:..=@@#..  ...:+@+.........\n");
    printf("............-+#+*@-#@@%+@%+@%....:=@@@:*%=#@*=@-#@@@===%%@@#.**=@++@@+:..:-:=@@#.. ....:+@+.........\n");
    printf("...........:*@*@%*@%#-#@*#@#+....:=@@@:*@%+@*-@-#@@@==@%+%@#-+#@@@+=*%#+:...+@@#.. ....:+@+.........\n");
    printf("..........:#%@@+*%@@%@@%:=@-.....:=@@@:*@@#@*=@-#@@@==@@%@@#:#*+@@+=%+#@-...=@@#.......:+@+.........\n");
    printf("........:%@@++@=+@@@@:...=@-.....:=@@@-*@%:#@%@-#@@@==@*@@@%%==@@@+=@=*@-...=@@#.:%....:=@+.........\n");
    printf(".......=@@@=*%.%*+@@@....=@-.....:=@@@@@@@@@*=@-#@@@==@@@@@#.%*:@=*%.#@@-...=@@#.-@:...:=@+.........\n");
    printf(".....-%%-%@@@@@@@@@+.....=@-.....:=@@@=#@**@*=@-#@@@++@@@@*=@==@@@+=@@@@-..:+@@#.-@:..-*#.+#:.......\n");
    printf(".....-%-%@-+@++%*@.......-%-:....-=@@@@@@@-#@@@-#@@=#@+*@@*=@==@@@+-@@@@-...+@@#.-@:..-*#.+%:.......\n");
    printf(".....-%%********+=... .....+*...-*@@@@@@@##%##%#@@*@##%%@@@#.@@@.#+..=%@*-..=@@#.-@:..=*#.-+........\n");
    printf(".....-@*........:-#=........-%%%%*+@@@@@@%%@%@@%@@@%@@#@@@@#.@@@.::...:-@+..=@@#.-@:..-*#...........\n");
    printf(".......=%-.......:@+............  -@@@-#-#@==-@@-*-%==@@@@@#.@@@@=.......+@-=@@%*=....-*#...........\n");
    printf("........:##........*%.............-@@@@@@%:%@=@@=#@@==@@:#@#.@@@@=.........@@@@%*=....=##...........\n");
    printf("..........-%+:.....-=................:#@@@@@@@#=@@@@==@*@@**@=#@....... ....=@@#.+@%=-%@*...........\n");
    printf("............+#-:.....:*%:.....:+#+@@#--#@@@@@@@@@@@@@@@@@@@@@@@@............=@@#.+@.+##-............\n");
    printf(".............:+*-.....-+-:.--.=+++++#@@#++*::+@=-------. *#--##*............=@@#.+@.................\n");
    printf("...............:+#:......-#=:.+#....:--#@@#:=@@@*#@@@@%: *@@@=-*............=@@#.+@.................\n");
    printf("..................@+.. ... ....:@@@@@%:*@@@-=@@@@@@@@@%: *@@@==@............=@@@@-..................\n");
    printf("...................+%.*@@#-%@@@@:.-@@@@@@#.:+@@@@@@@@@#::*%@@==@............=@@@%-..................\n");
    printf(".................... .-:*-%@@@@@@#+@@@%%@%=%@%---------=@@*+%++@............=@@#....................\n");
    printf("..................... ...:=%@@@@@@@@@@:=@@@@#*@@@@@@@@@@@@@%=@@@............=@@#....................\n");
    printf("........................:+@@#=*+@*-@#-%@+*@+*%-%+*@*-%%+%@@@@@@@............=@@#....................\n");
    printf(".......................:%@@#%@%%%***#*@@@@@@@@@@@@@@@@@@@@@@@@@@............=@@#....................\n");
    printf("......................=%@@@%%##%@%@.*@@%%@@*@@@#%%#@@@*@*%@%%@@@............=@@#....................\n");
    printf("......................+@@@@@*%@@@@@*%@@@%##@#@*@@@%#%*@@*%@@##@@............=@@#....................\n");*/
    (void)halt(255);
    } 
}

