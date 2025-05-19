#pragma once

#ifndef PAGING_H
#define PAGING_H
// #ifndef ASM
#define VIDEO_MEM 0xB8000
#define KERNEL_MEM 0x400000
#define ALIGN_OFF 4096
#include "x86_desc.h"
#include "types.h"
#include "lib.h"


typedef struct page_dir_4kb_t
{
    union{
        uint32_t entry;
        struct
        {
            uint32_t p : 1;// present
            uint32_t rw : 1;// read/write
            uint32_t us : 1;// user/supervisor mode - permission mode
            uint32_t pwt : 1;// page write-though
            uint32_t pcd : 1;// page cached
            uint32_t a : 1;// accessed - 1: Yes, 0: No
            uint32_t reserved : 1;
            uint32_t ps : 1;// page size - 1: 4mb, 0: 4kb
            uint32_t g : 1;
            uint32_t avl : 3; // extra bits ?
            uint32_t offset_32_12: 20; // PDE table location           
        } __attribute__((packed));
    };
    
} page_dir_4kb_t; // structure for page directories

typedef struct page_dir_4mb_t
{
    union{
        uint32_t entry;
        struct
        {
            uint32_t p : 1;       // Present
            uint32_t rw : 1;      // read/write
            uint32_t us : 1;      // user/supervisor mode - permission mode
            uint32_t pwt : 1;     // page write-though
            uint32_t pcd : 1;     // page cached
            uint32_t a : 1;       // accessed - 1: Yes, 0: No
            uint32_t d : 1;       // dirty
            uint32_t ps : 1;      // page size - 1: 4mb, 0: 4kb
            uint32_t g : 1;       // global
            uint8_t avl : 3;      // extra bits ?
            uint32_t pat : 1;     // Page Attribute table
            uint32_t reserved : 9;
            uint32_t base_addr : 10; // PDE table location
        } __attribute__((packed));
    };
    
} page_dir_4mb_t; // structure for page directories

uint32_t page_dir[1024] __attribute__((aligned(4096))); // page directory is aligned to 4kb

typedef struct page_tab_t
{
    union{
        uint32_t entry;
        struct
        {
            uint32_t p : 1;
            uint32_t rw : 1;
            uint32_t us : 1;
            uint32_t pwt : 1;
            uint32_t pcd : 1;
            uint32_t a : 1;
            uint32_t d : 1;
            uint32_t pat : 1;
            uint32_t g : 1;
            uint8_t avl : 3;
            uint32_t offset_32_12 : 20;
            
        } __attribute__((packed));
    };
    
} page_tab_t; // structure for page tables
uint32_t first_page_tab[1024] __attribute__((aligned(4096)));
//page_tab_t first_page_tab[1024] __attribute__((aligned(4096)));
//page table for video memory
uint32_t vidmap_page_tab[1024] __attribute__((aligned(4096)));
// page_tab_t sec_page_tab[1024] __attribute__((aligned(4096))); // page tables are aligned to 4kb
// extern x86_desc_t cr0_mask;
void paging_init();




// #endif // asm
#endif // paging_h

