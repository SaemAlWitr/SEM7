#include<stdio.h>
#include<stdlib.h>
#include<string.h>

void trim(char *s) {
    if (!s) return;
    char *start = s;
    while (*start && *start == ' ') start++;
    if (start != s) memmove(s, start, strlen(start) + 1);

    size_t len = strlen(s);
    while (len > 0 && s[len-1] == ' ') {
        s[len - 1] = '\0';
        len--;
    }
}

char** seq_tokenize(char* _cmd){
	int l = 0;
	int ct = 1;
	char* cmd = strdup(_cmd);
	while(cmd[l]!='\0'){
		if(cmd[l] == ';') ct++;
		l++;
	}
	char** tokens = malloc((ct+1) * sizeof(char*));
	char* tok = strtok(cmd, ";");
	l = 0;
	while(tok){
		trim(tok);
		char* _tok = strdup(tok);
		tokens[l++] = _tok;
		tok = strtok(NULL,";");
	}
	tokens[l] = NULL;
	free(cmd);
	return tokens;
}

int main(){
    char* cmd = strdup("cat fil1.txt");
    char** toks = seq_tokenize(cmd); int i =0;
    while(toks[i]){
        printf("%s\n",toks[i++]);
    }
    free(cmd);
    i = 0;
    while(toks[i]){
        free(toks[i++]);
    }
    free(toks);
}