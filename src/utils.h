#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper function to trim leading and trailing whitespace.
inline char* trim_whitespace(char* str){
    char* end;
    while(*str == ' ' || *str == '\t' ||  *str == '\n' || *str == '\r') str++;
    if(*str == 0) return str;
    end = str + strlen(str) - 1;
    while(end > str && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) end--;
    *(end + 1) = 0;
    return str;
}

// Helper function to parse an integer value from a YAML line.
inline int parse_int_value(const char* line){
    while (*line && *line != ':') line++;
    if (*line == ':') line++;
    return atoi(trim_whitespace((char*)line));
}

// Helper function to print the progress bar
inline void print_progress(int step, int total_steps, float step_avg_loss){
    int bar_width = 50; // Width of the progress bar
    int pos = (step * bar_width)/total_steps;

    printf("[");
    for(int i=0; i<bar_width; ++i){
        if(i<pos){
            printf("=");
        } else if(i==pos){
            printf(">");
        } else{
            printf(" ");
        }
    }

    printf("] %d%% | Step Average Loss: %.2f\r", (step * 100) / total_steps, step_avg_loss);
    fflush(stdout);
}