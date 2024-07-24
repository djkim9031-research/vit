#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper function to trim leading and trailing whitespace.
char* trim_whitespace(char* str){
    char* end;
    while(*str == ' ' || *str == '\t' ||  *str == '\n' || *str == '\r') str++;
    if(*str == 0) return str;
    end = str + strlen(str) - 1;
    while(end > str && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) end--;
    *(end + 1) = 0;
    return str;
}

// Helper function to parse an integer value from a YAML line.
int parse_int_value(const char* line){
    while (*line && *line != ':') line++;
    if (*line == ':') line++;
    return atoi(trim_whitespace((char*)line));
}