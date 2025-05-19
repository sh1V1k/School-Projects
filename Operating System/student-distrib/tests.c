/*
#include "tests.h"
#include "x86_desc.h"
#include "lib.h"
#include "rtc.h"
#include "keyboard.h"
#include "idt.h"
#include "paging.h"
#include "file_system.h"
#include "syscall.h"

#define PASS 1
#define FAIL 0

 format these macros as you see fit 
#define TEST_HEADER 	\
	printf("[TEST %s] Running %s at %s:%d\n", __FUNCTION__, __FUNCTION__, __FILE__, __LINE__)
#define TEST_OUTPUT(name, result)	\
	printf("[TEST %s] Result = %s\n", name, (result) ? "PASS" : "FAIL");

static inline void assertion_failure(){
	 Use exception #15 for assertions, otherwise
	   reserved by Intel 
	asm volatile("int $15");
}

//test.c just got modifiedgit #2

 Checkpoint 1 tests 

 IDT Test - Example
 * 
 * Asserts that first 10 IDT entries are not NULL
 * Inputs: None
 * Outputs: PASS/FAIL
 * Side Effects: None
 * Coverage: Load IDT, IDT definition
 * Files: x86_desc.h/S
 
int idt_test(){
	TEST_HEADER;

	int i;
	int result = PASS;
	for (i = 0; i < 10; ++i){
		if ((idt[i].offset_15_00 == NULL) && 
			(idt[i].offset_31_16 == NULL)){
			assertion_failure();
			result = FAIL;
		}
	}

	return result;
}
 Divide by Zero test
 * 
 * Throws exception due to divide by 0.
 * Inputs: None
 * Outputs: PASS/FAIL
 * Side Effects: None
 * Coverage: Load IDT, IDT definition
 * Files: x86_desc.h/S
 
int div_0_test(){
	TEST_HEADER;

	int zero =0;
	int sixnine = 69;
	sixnine /=zero;
	return FAIL;//shouldn't get here
}
 Null pointer test
 * 
 * Throws exception due to llbnull pointer.
 * Inputs: None
 * Outputs: PASS/FAIL
 * Side Effects: None
 * Coverage: Load IDT, IDT definition
 * Files: x86_desc.h/S
 
int null_ref(){
	TEST_HEADER;

	int* test = NULL;
	int testout;
	testout = *test;
	return FAIL;
}

 Trigger any interrupt
 * 
 * Throws exception as specified
 * Inputs: None
 * Outputs: PASS/FAIL
 * Side Effects: None
 * Coverage: Load IDT, IDT definition
 * Files: x86_desc.h/S
 
int trigger_five(){
	TEST_HEADER; 

	// Attempt to trigger the exception #5 -- should work for any 
	asm volatile("      \n\
        movl $6, %eax 	\n\
        int $0x80 		\n\
    "
    );
	// asm volatile("int $0x5");

	return FAIL; 
}

 RTC Test
 * 
 * Validates entry into RTC.
 * Inputs: None
 * Outputs: PASS/FAIL
 * Side Effects: None
 * Coverage: Load RTC
 * Files: rtc.c/h
 
int rtc_ticks(){
	TEST_HEADER;

	// after recieving number of rtc interrupts, continue
	while (rtc_count < 42){
	}

	return PASS;
}

 Out of Bounds paging error
 * 
 * Validates error thrown when Out-of-Bounds paging occurs
 * Inputs: None
 * Outputs: PASS/FAIL
 * Side Effects: None
 * Coverage: Load IDT, IDT definition
 * Files: paging.c/h, paging_asm.c/h

int oob_page_ref() {
	TEST_HEADER;

	int test;
	int *oob_adr = (int*)(0x800000+ 15);
	test = *oob_adr;

	return FAIL;
}
 Paging entry
 * 
 * Validates video memory and paging is accessible
 * Inputs: None
 * Outputs: PASS/FAIL
 * Side Effects: None
 * Coverage: Load IDT, IDT definition
 * Files: paging.c/h, paging_asm.c/h
 
int paging_test(){
	TEST_HEADER;

	int a;
	int* ptr;
	// get video memory ptr

	ptr = (int*)(VIDEO_MEM);
	// get value at video memory base
	a= *ptr;

	// get kernel memory ptr
	ptr = (int*)(KERNEL_MEM);
	//get value at kernel memory ptr
	a = *ptr;

	return PASS;
}


 Paging non-empty
 * 
 * Make sure page directory first and second are none-zero and page video entry for
 * video memory is none-zero.
 * Inputs: None
 * Outputs: PASS/FAIL
 * Side Effects: None
 * Coverage: Load IDT, IDT definition
 * Files: paging.c/h, paging_asm.c/h
 
int paging_nonempty_test(){
	TEST_HEADER;

	int result = PASS;
	if((page_dir[0] == 0 || page_dir[1] == 0) || first_page_tab[0xb8] == 0){
		assertion_failure();
		result = FAIL;
	}

	return result;
	
}
 Checkpoint 2 tests 

int terminal_test(){
	TEST_HEADER;
	int result = PASS;
	int i = 1;
	int j = 0;
	unsigned number_chars_printed = 0;
	char temp_buf[2*BUFFER_SIZE];
	for(j = 0; j < 2*BUFFER_SIZE; j++){
		temp_buf[j] = 'a';
	}
	while(i < 2*BUFFER_SIZE){
		read_terminal(0, temp_buf, i);
		number_chars_printed = write_terminal(0, NULL, i);
		if(i < BUFFER_SIZE && (number_chars_printed != i+1)){
			assertion_failure();
			result = FAIL;
		}
		else if(i >= BUFFER_SIZE && (number_chars_printed != BUFFER_SIZE)){
			assertion_failure();
			result = FAIL;
		}
		i++;
	}
	return result;
}

 File Read Test
 * 
 * Checks to see if we fill the buffer with the correct contents of the associated file
 * Inputs: None
 * Outputs: PASS/FAIL
 * Side Effects: None
 * Coverage: file_read
 * Files: file_system.c/h
 
int filesystem_test(){
	TEST_HEADER;

	 Magic Number for the associated files and what you have to pass into file read
	grep: index = 3, file size = 6149         executable file
	frame0.txt: index = 10, file size = 187   regular file
	fish: index = 6, file size = 36164        large file
	
	int result = PASS;
	char buffer[5605]; //! MAGIC NUMBER, number of characters including newline
	if(file_read(2, buffer, 5605) == 0){
		write_buffer_to_screen(buffer, 5605);
	} else { result = FAIL; }
	return result;
}

RTC Virtualization Test
 * 
 * Checks to see if we virtualized our rtc correctly
 * Inputs: None
 * Outputs: PASS/FAIL
 * Side Effects: None
 * Coverage: rtc_open, rtc_write, rtc_read
 * Files: rtc.c/h

int virt_rtc_test(){
	TEST_HEADER;
	int result = PASS;
	int i;
	int count = 0;
	uint32_t rate;
	void* ptr;
	uint32_t rates_to_test[5] = {64, 128, 256, 512, 1024};

	for(i = 0; i < 5; i++){
		rate = rates_to_test[i];
		printf("rate: %d \n", rate);
		ptr = &rate;
		(void)rtc_write(0, ptr, 0); //new rate
		while(count < 50){
			// test for 3 seconds
			if(rtc_read(0, ptr, 0) == 0){ putc('1'); }
			count++;
		}
		count = 0;
		clear();
		(void)rtc_write(0, ptr, 0); //new rate
	}

	(void)rtc_open((uint8_t*)rates_to_test); //initialization to 2
	printf("RTC open test, rate: 2 \n");
	while(count < 10){
		if(rtc_read(0, ptr, 0) == 0){ putc('1'); }
		count++;
	}

	return result;
}

 Directory read test
 * 
 * Validates that directory file fills up the buffer correctly
 * Inputs: None
 * Outputs: PASS/FAIL
 * Side Effects: None
 * Coverage: directory_read
 * Files: file_system.c/h

int dir_read_test(){
	TEST_HEADER;
	int result = PASS;
	char buffer[33]; //! MAGIC NUMBER, number of characters including newline
	int32_t i;
	for(i = 0; i < boot_block->dir_entries; i++){
		if(dir_read(i, buffer, 32) == 0){
			buffer[33] = '\n'; //put new line char at the end of the buffer so the next file in printed on a separate line in terminal
			write_buffer_to_screen(buffer, 34);
		} else { result = FAIL; }
	}
	return result;
}

 Checkpoint 3 tests 
 Checkpoint 4 tests 
 Checkpoint 5 tests 


 Test suite entry point 
void launch_tests(){
	//TEST_OUTPUT("idt_test", idt_test());
	//TEST_OUTPUT("divide error", div_0_test());
	//TEST_OUTPUT("throw 5", trigger_five());
	//TEST_OUTPUT("rtc ticks", rtc_ticks());
	//TEST_OUTPUT("null ref", null_ref());
	//TEST_OUTPUT("paging oob", oob_page_ref());
	//TEST_OUTPUT("paging test", paging_test());
	//TEST_OUTPUT("terminal test", terminal_test());
	//TEST_OUTPUT("file system test", filesystem_test());
	//TEST_OUTPUT("virtualization rtc test", virt_rtc_test());
	//TEST_OUTPUT("directory read test", dir_read_test());
}
*/
