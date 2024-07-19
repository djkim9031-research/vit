#pragma once
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

struct BGR{
    // bmp color channel order
    uint8_t blue;
    uint8_t green;
    uint8_t red;
};

// BMP image file reader function
//
// @param filename      path to the .bmp image file.
// @param width         width of the bmp image.
// @param height        height of the bmp image.
// @param pixels        pixels extracted from the bmp image.
//
inline int BMPReader(const char* filename, const int width, const int height, BGR** pixels){
    FILE* file = fopen(filename, "rb");
    if(!file){
        perror("Unable to open the file");
        return -1;
    }

    // 4-byte alignment of BMP.
    // For each row, 24-bit (corr to pixels) + extra 3 bytes padding 
    // to account for any remainders
    // With ~3 = binary(00), current row bytes are rounded up 
    // to the nearest multiple of 4-byte (e.g., 5 pixels row = 15 bytes => aligned to 16 bytes)
    int row_padded = (width * 3 + 3) & (~3);
    fseek(file, 54, SEEK_SET); // Skip the header (54 bytes)
    
    *pixels = (BGR*)malloc(width*height*sizeof(BGR));
    if(*pixels == NULL){
        perror("Unable to allocate memory");
        fclose(file);
        return -1;
    }

    for(int y=0; y<height; ++y){
        for(int x=0; x<width; ++x){
            // BMPs are stored bottom-to-top.
            fread(&(*pixels)[(height - y - 1) * width + x], sizeof(BGR), 1, file);
        }
        fseek(file, row_padded - width * 3, SEEK_CUR);
    }

    fclose(file);
    return 0;
}


inline int ReadAllBMPsInDirectory(const char* dirPath, BGR*** allPixels, int& nImages, int width, int height){
    DIR* dir = opendir(dirPath);
    if(!dir){
        perror("Unable to open the given directory.");
        return -1;
    }

    struct dirent* entry;
    nImages = 0;
    while((entry = readdir(dir)) != NULL){
        if(entry->d_type == DT_REG && strstr(entry->d_name, ".bmp")){
            nImages++;
        }
    }
    rewinddir(dir); // Rewind the directory pointer to read all bmp images.

    *allPixels = (BGR**)malloc(nImages*sizeof(BGR*));
    if(*allPixels == NULL){
        perror("Unable to allocate memory");
        closedir(dir);
        return -1;
    }

    int idx = 0;
    while((entry = readdir(dir)) != NULL){
        if (entry->d_type == DT_REG && strstr(entry->d_name, ".bmp")){
            char filePath[1024];
            snprintf(filePath, sizeof(filePath), "%s/%s", dirPath, entry->d_name);

            BGR* pixels = NULL;
            // Read the BMP file to get pixel data
            if (BMPReader(filePath, width, height, &pixels) == 0){
                (*allPixels)[idx] = pixels;
                idx++;
            } else{
                printf("Image %d failed to read pixels", idx);
                free(pixels);
            }
        }
    }

    closedir(dir);
    return 0;
}


