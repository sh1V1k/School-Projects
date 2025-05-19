#pragma once
#ifndef FILE_SYS_H
#define FILE_SYS_H

#include "types.h"

#define FILE_SEGMENT 4096 //4kb 
#define DIR_SIZE 64 //64b in dir 
#define MAX_FILES DIR_SIZE-1 //might err - need to be 63
#define FILE_NAME_LEN 32 //32b - max 32 char len

#define MAX_FILE_OPEN 8

typedef struct dentry{ //standard entry in the file system
  int8_t filename[FILE_NAME_LEN];
  int32_t filetype; //File types are 0 for a file giving user-level access to 
                    //the real-time clock (RTC), 1 for the directory, and 2 for a regular file.
  int32_t inode_num; //only fr reg files, not dir/rtc
  int8_t reserved[24]; 
}dentry_t;

typedef struct boot{ //init block in file sys 
  int32_t dir_entries;
  int32_t num_inodes;
  int32_t num_data_block;
  int8_t reserved[52];
  dentry_t dir_reserved[64]; //used be 63 for "."
}boot_t;

typedef struct inode{
  int32_t data_len; //size of that associated file
  int32_t data_block[1023]; //index of datablock in datablock list
}inode_t;

typedef struct file_operation_file{ //jump table for function pointers
  int32_t (*file_open)(const uint8_t *fname);
  int32_t (*file_read)(int32_t fd, void *buf, int32_t nbytes);
  int32_t (*file_close)(int32_t fd); 
  int32_t (*file_write)(int32_t fd, const void *buf, int32_t nbytes);
}fof_t;

typedef struct file_operation_dir{ //jump table for function pointers
  int32_t (*dir_open)(const uint8_t *fname);
  int32_t (*dir_read)(int32_t fd, void *buf, int32_t nbytes);
  int32_t (*dir_close)(int32_t fd); 
  int32_t (*dir_write)(int32_t fd, const void *buf, int32_t nbytes);
}fod_t;

typedef struct block_data{ //data block descriptor 
  uint8_t data[4096]; // each datablock has 4kb of data stored in it
}block_data_t;


inode_t* first_inode; 
boot_t* boot_block;
dentry_t* first_dir_entry;
block_data_t* first_data_block;
uint32_t directory_entry;  


void file_system_init(uint32_t start_adr);

/*Find the file with name given as the input in the file system. 
 Copy the file name, file type and inode number into the dentry object given as input. */
int32_t read_dentry_by_name (const uint8_t* fname, dentry_t* dentry);
/*Find the file with index given as the input in the file system. 
 Copy the file name, file type and inode number into the dentry object given as input. */
int32_t read_dentry_by_index (uint32_t index, dentry_t* dentry);

/*  read data from file given inode number. */
int32_t read_data(uint32_t inode, uint32_t offset, uint8_t* buf, uint32_t length);

//init file structure at location by name, ret 0 
int32_t file_open(const uint8_t* fname); //return inode index
//undoes open func, ret 0 does nothing?
int32_t file_close(int32_t fd);

//reads nbytes from fd (by location?) into buuf
int32_t file_read(int32_t fd, void* buf, int32_t nbytes); //make to update file pos for the fd array in the future

//populates fd with contnts of buf, nbytes = 
int32_t file_write(int32_t fd, const void* buf, int32_t nbytes);

//calls read denttry_by name - opens dir filetype
int32_t dir_open(const uint8_t* fname);

/* does nothing*/
int32_t dir_close(int32_t fd);

// returns -1 since we have a read only file system
int32_t dir_write(int32_t fd, const void* buf, int32_t nbytes);

//stores the filename of the directory entry index passed in through fd
int32_t dir_read(int32_t fd, void* buf, int32_t nbytes); //make to update file pos for the fd array in the future

//print contents of buffer to screen, helper to test our functions
void write_buffer_to_screen(const void* buf, int32_t nbytes);


#endif
