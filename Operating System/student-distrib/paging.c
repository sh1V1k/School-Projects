#include "paging.h"
#include "paging_asm.h"



/* void paging_init()
 *
 * initalizes paging
 * Inputs: None
 * Outputs: Void
 * Side Effects: paging activated
 */
void paging_init(){ 
    printf("Initializing Page Directory\n");
    int i;
    for(i = 0; i < 1024; i++){
        page_dir[i] = 0;
        first_page_tab[i] = 0;
    }
    
    page_dir_4kb_t first_entry;
    first_entry.p = 1;
    first_entry.rw = 1;
    first_entry.us = 0;
    first_entry.pwt = 0;
    first_entry.pcd = 0;
    first_entry.a = 0;
    first_entry.ps = 0;
    first_entry.avl = 0; // set all values for 4kb pages for first 4mb of memory
    first_entry.g = 0;
    first_entry.reserved = 0;
    first_entry.offset_32_12 = (uint32_t)(first_page_tab) >> 12; // get the 20 bits needed by shifting 32bits 12 to right
    page_dir[0] = (uint32_t)(first_entry.entry);

    
    
    page_dir_4mb_t second_entry;
    second_entry.avl = 0; // extra bits ?
    second_entry.ps = 1;  // page size - 1: 4mb, 0: 4kb
    second_entry.a = 0;   // accessed - 1: Yes, 0: No
    second_entry.pcd = 0; // page cached
    second_entry.pwt = 0; // page write-though
    second_entry.us = 0;  // user/supervisor mode - permission mode
    second_entry.rw = 1;  // read/write
    second_entry.p = 1;   // Present
    second_entry.pat = 0; // Page Attribute table
    second_entry.d = 0;   // dirty
    second_entry.g = 0;   // global
    //second_entry.reserved = 0;
    second_entry.base_addr = KERNEL_MEM >> 22; //kernel address right shifted by 12
    page_dir[KERNEL_MEM >> 22] = (uint32_t)(second_entry.entry);

    page_tab_t vid;
    vid.p = 1;
    vid.rw = 1;
    vid.us = 0;
    vid.avl = 0;
    vid.g = 0;
    vid.pat = 0;
    vid.d = 0;
    vid.a = 0;
    vid.pcd = 0;
    vid.pwt = 0;
    vid.offset_32_12 = 0xb8; //0xb8 * 4096 = 0xb8000 which is video memory
    first_page_tab[0xb8] = (uint32_t)vid.entry;

    //Terminal 1 (idx 0)
    vid.offset_32_12 = 0xb9; 
    first_page_tab[0xb9] = (uint32_t)vid.entry;

    //Terminal 2 (idx 1)
    vid.offset_32_12 = 0xba; 
    first_page_tab[0xba] = (uint32_t)vid.entry;

    //Terminal 3 (idx 2)
    vid.offset_32_12 = 0xbb; 
    first_page_tab[0xbb] = (uint32_t)vid.entry;

    vid.offset_32_12 = 0xb8; 
    first_page_tab[0xbc] = (uint32_t)vid.entry;

    // vid.offset_32_12 = 0xba; 
    // first_page_tab[0xbd] = (uint32_t)vid.entry;

    // vid.offset_32_12 = 0xbb; 
    // first_page_tab[0xbe] = (uint32_t)vid.entry;

    enable_paging((uint32_t)page_dir);

    //printf("done init Page Directory\n");
}
