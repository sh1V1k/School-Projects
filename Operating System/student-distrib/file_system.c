#include "file_system.h"
#include "lib.h"
#include "rtc.h"
#include "syscall.h"
// // TODO : all of these methods need param checking

/* Initialize our file system by storing the memory address*/
/* file_system_init(uint32_t start_adr)
 *
 * Save useful memory address, like the boot block, first inode, and first data block
 * Inputs: start address of that module, in out case this is module 0
 * Outputs: NONE
 * Side Effects: Sets the global varibales that we use in our file read function
 */
void file_system_init(uint32_t start_adr)
{
    printf("starting file system initialization \n");
    printf("module 0 address: %d \n", start_adr); // sanity check
    boot_block = (boot_t *)start_adr;
    first_inode = (inode_t *)((uint32_t)boot_block + FILE_SEGMENT);
    first_dir_entry = boot_block->dir_reserved;
    directory_entry = 0;
    // start of data blocks
    first_data_block = (block_data_t *)((uint32_t)first_inode + (FILE_SEGMENT) * (boot_block->num_inodes));
    printf("done with file system initialization \n");
    // data_blocks start when inodes end
}

/* Read Dentry By Name */
/* read_dentry_by_name(const uint8_t *fname, dentry_t *dentry)
 *
 * Copy the file name, file type and inode number into the dentry object given as input.
 *
 * Inputs: The name of the file we want to find and copy, and the dentry varible that we want to fill with the
 *         associated values
 * Outputs: 0 on success, -1 on failure
 * Side Effects: NONE
 */
int32_t read_dentry_by_name(const uint8_t *fname, dentry_t *dentry)
{
    int dir_num = boot_block->dir_entries;
    int i;
    dentry_t cur_dentry;

    // iterate through directory entries
    for (i = 0; i < dir_num; i++)
    { // i = i since first directory is "."
        cur_dentry = boot_block->dir_reserved[i];
        if (strncmp((const int8_t *)cur_dentry.filename, (const int8_t *)fname, (uint32_t)FILE_NAME_LEN) == 0)
        {                                                                        // found file
            (void)strncpy(dentry->filename, cur_dentry.filename, FILE_NAME_LEN); // only write the first 32 character of the file name
            dentry->filetype = cur_dentry.filetype;
            dentry->inode_num = cur_dentry.inode_num;
            return 0;
        }
    }
    return -1;
}

/* Read Dentry By Index */
/* read_dentry_by_index(uint32_t index, dentry_t *dentry)
 *
 * Copy the file name, file type and inode number into the dentry object given as input.
 *
 * Inputs: The index in the directory entries array, and the dentry varible that we want to fill with the
 *         associated values
 * Outputs: 0 on success, -1 on failure
 * Side Effects: NONE
 */
int32_t read_dentry_by_index(uint32_t index, dentry_t *dentry)
{
    dentry_t cur_dentry;
    if (index > boot_block->dir_entries)
    {
        return -1;
    }
    else
    {
        cur_dentry = boot_block->dir_reserved[index];
        (void)strncpy(dentry->filename, cur_dentry.filename, FILE_NAME_LEN); // only write the first 32 character of the file name
        dentry->filetype = cur_dentry.filetype;
        dentry->inode_num = cur_dentry.inode_num;
        return 0;
    }
}

/* Read Data, helper function used by file read */
/* read_data(uint32_t inode, uint32_t offset, uint8_t *buf, uint32_t length)
 *
 * Inputs: The inode number of the associated file, the offset that we want to read from within the file, the
 *         buffer we want to write to, and the length of the file in bytes (file size). Fffset acts like cursor in data being read
 * Outputs: number of bytes not read, so 0 on success
 * Side Effects: overwrites old buffer values
 */
int32_t read_data(uint32_t inode, uint32_t offset, uint8_t *buf, uint32_t length)
{

    if (inode >= boot_block->num_inodes || inode < 0 || buf == NULL)
    {
        return -1;
    }

    inode_t *cur_inode = (inode_t *)((uint32_t)first_inode + (FILE_SEGMENT)*inode);

    uint32_t data_block_index = offset / (FILE_SEGMENT);
    uint32_t cursor = offset % (FILE_SEGMENT);
    uint32_t cur_block;
    uint32_t bytes_read = 0;
    uint32_t i;

    for (i = 0; i < length; i++, cursor++, bytes_read++) // len in terms of bytes cursor = num bytes
    {
        if ((i + offset) >= cur_inode->data_len)
        {
            return bytes_read;
        } // end condition

        if (cursor / FILE_SEGMENT > 0)
        {
            cursor = 0;
            data_block_index++;
        } // switch data blocks

        cur_block = cur_inode->data_block[data_block_index];

        if (cur_block >= boot_block->num_data_block)
        {
            return -1;
        }
        uint8_t * temp = (uint8_t *)((uint32_t)first_data_block + cur_block*(FILE_SEGMENT) + cursor);
        buf[i] = *temp;

    }
    return bytes_read; // should be 0 if it makes it out bc copied all the elements in the len
}

/* File Read */
/* file_read(int32_t fd, void *buf, int32_t nbytes)
 *
 * Inputs: the index in the directory entries array stored in boot block, the buffer we want to write to
 * and the number of bytes we want to write to that buffer. Uses a helper function called read data
 * Outputs: 0 on success, -1 on failure
 * Side Effects: overwrites old buffer values
 */
int32_t file_read(int32_t fd, void *buf, int32_t nbytes)
{

    //! currently which file im accessing via fd and open_file_idx is 100% fucked
    pcb_t *cur_pcb = get_pcb(); //gets pcb without needing index

    

    //(void)read_dentry_by_name((const uint8_t *)buf, &cur_dentry); // return -1

    if (cur_pcb->open_files[fd].f_filetype != 2)
    {
        return -1;
    } // not a regular file type so return error


    if ( !cur_pcb->open_files[fd].f_flags){ //if flags are off 
        return -1;
    }
    

    int32_t read_data_output = read_data(cur_pcb->open_files[fd].f_inode, cur_pcb->open_files[fd].f_pos, buf, nbytes);

    cur_pcb->open_files[fd].f_pos += read_data_output;

    return read_data_output > 0 ? read_data_output : 0;
}

/* does nothing for now*/
int32_t file_open(const uint8_t *fname)
{
    // LATER CHECKPOINT STUFF BUGGY AS HELL
    // dentry_t cur_dentry;
    // fds_t opened_file;
    // fo_t* operations;
    // int file_type_flag;
    // if(open_file_idx > 7){ return -1; }//no more space, array is full
    // if(read_dentry_by_name(fname, &cur_dentry) == 0)
    // {
    //     file_type_flag = cur_dentry.filetype;
    //     if(file_type_flag == 1){ //directory file type
    //         operations->close = &dir_close;
    //         operations->open = &dir_open;
    //         operations->read = &dir_read;
    //         operations->write = &dir_write;
    //         opened_file.f_inode = 0;
    //     } else if(file_type_flag == 2){ //reg file type
    //         operations->close = &file_close;
    //         operations->open = &file_open;
    //         operations->read = &file_read;
    //         operations->write = &file_write;
    //         opened_file.f_inode = cur_dentry.inode_num;
    //     } else if(file_type_flag == 0){ //rtc file type
    //         operations->close = &rtc_close;
    //         operations->open = &rtc_open;
    //         operations->read = &rtc_read;
    //         operations->write = &rtc_write;
    //         opened_file.f_inode = 0;
    //     }
    //     opened_file.file_opertation = operations;
    //     opened_file.f_flags = 1;
    //     open_files[open_file_idx] = opened_file;
    //     //find next open space if there is one
    //     while(open_file_idx != 8 && open_files[open_file_idx].f_flags == 1){ //what happens in the case that nothing was ever opended there?
    //         open_file_idx++;
    //     }
    //     return 0;
    // }
    // else
    // {
    //     return -1; // returns -1 if fname doesn't exist
    // }
    return 0;
}

/* File Write */
/* file_write(int32_t fd, const void *buf, int32_t nbytes)
 * Inputs: the index in the directory entries array stored in boot block, the buffer we want to write to
 * and the number of bytes we want to write to that buffer
 * Side Effects: writes to the file
 */
// returns an error since we have a read only file system
int32_t file_write(int32_t fd, const void *buf, int32_t nbytes)
{
    return -1;
}

/* File Close */
/* file_close(int32_t fd)
 * Inputs: the index in the directory entries array stored in boot block
 * Side Effects: closes the file
 */
/* does nothing for now*/
int32_t file_close(int32_t fd)
{
    // LATER CHECKPOINT STUFF BUGGY AS HELL
    // if(fd < 0 || fd > 7){ return -1; } //bounds check
    // if(open_files[fd].f_flags == 0){ return 0; }// does nothing the file is not in use
    // else{//file in use and we want to close it
    //     open_files[fd].f_flags = 0; //set to not in use and update idx
    //     open_file_idx = fd;
    // }
    return 0;
}

/* Directory Open */
/* dir_open(int32_t fd)
 * Inputs: the directory to be opened
 * Side Effects: opens the directory
 */

/* does nothing for now*/
int32_t dir_open(const uint8_t *fname)
{
    // use read_dentry_by_name
    // dentry_t cur_dentry;
    // fds_t opened_file;
    // if (open_file_idx > 7)
    // {
    //     return -1;
    // } // no more space, array is full
    // if (read_dentry_by_name(fname, &cur_dentry) == 0)
    // {
    //     opened_file.file_opertation->close = dir_close;
    //     opened_file.file_opertation->open = dir_open;
    //     opened_file.file_opertation->read = dir_read;
    //     opened_file.file_opertation->write = dir_write;
    //     opened_file.f_inode = 0;
    //     opened_file.f_flags = 1;
    //     open_files[open_file_idx] = opened_file;
    //     // find next open space if there is one
    //     while (open_file_idx != 8 && open_files[open_file_idx].f_flags == 1)
    //     { // what happens in the case that nothing was ever opended there?
    //         open_file_idx++;
    //     }
    //     return 0;
    // }
    // else
    // {
    //     return -1;
    // }
    return 0;
}
/* Directory Close */
/* dir_close(int32_t fd)
 * Inputs: the index in the directory entries array stored in boot block
 * Side Effects: closes the directory
 */

/* does nothing for now*/
int32_t dir_close(int32_t fd)
{
    // FUTURE CHECKPOINT STUFF
    //  if (fd < 0 || fd > 7)
    //  {
    //      return -1;
    //  } // bounds check
    //  if (open_files[fd].f_flags == 0)
    //  {
    //      return 0;
    //  } // does nothing the file is not in use
    //  else
    //  {                               // file in use and we want to close it
    //      open_files[fd].f_flags = 0; // set to not in use and update idx
    //      open_file_idx = fd;
    //  }
    return 0;
}

/* Directory Write */
/* dir_write(int32_t fd, const void *buf, int32_t nbytes)
 * Inputs: the index in the directory entries array stored in boot block, the buffer we want to write to
 * and the number of bytes we want to write to that buffer
 * Side Effects: writes to the file
 */

// returns an error since we have a read only file system
int32_t dir_write(int32_t fd, const void *buf, int32_t nbytes)
{
    return -1;
}

/* Directory Read */
/* dir_read(int32_t fd, void *buf, int32_t nbytes)
 *
 * Inputs: the index in the directory entries array stored in boot block, the buffer we want to write to
 * and the number of bytes we want to write to that buffer
 * Outputs: 0 on success, -1 on failure
 * Side Effects: overwrites old buffer values
 */
int32_t dir_read(int32_t fd, void *buf, int32_t nbytes)
{ // look at syscall ls, 16
    dentry_t cur_dentry;
    if (directory_entry > 16)
    {
        directory_entry = 0;
        return 0;
    }
    if (read_dentry_by_index(directory_entry, &cur_dentry) == 0)
    {
        (void)strncpy((int8_t *)buf, cur_dentry.filename, nbytes);
        directory_entry++;
        return nbytes;
    }
    else
    {
        return -1;
    }
}

/* Helper function that prints a buffer to the terminal */
/* write_buffer_to_screen(const void *buf, int32_t nbytes)
 *
 * Inputs: the buffer we want to print and the amount of bytes we want to print from that buffer
 * Outputs: NONE
 * Side Effects: overwrites old buffer values
 */
void write_buffer_to_screen(const void *buf, int32_t nbytes)
{
    int32_t i;
    for (i = 0; i < nbytes; i++)
    {
        putc(*((uint8_t *)buf + i)); // pointer array duality
    }
}
