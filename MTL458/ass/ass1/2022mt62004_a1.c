#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include <stdbool.h>
#include <glob.h>
#include <termios.h>
#include <ctype.h>
#include <sys/types.h>  // pid_t, ssize_t (portability)
#include <sys/stat.h>   // mode constants for open
#define ERROR fprintf(stderr, "Invalid Command\n")

const int MAX_HISTORY = 2048;
const int MAX_CMD_LEN = 2048;
const int MAX_ARGC = 100;
/*
THINGS TO BE DONE
* implement history and history n


* print MTL458 > 
* take input 
* if tab is pressed, try to auto-complete
* do wildcard completion in the text if * is found in the input
* while doing the following, for each command check if it is valid (eg. lss invalid)
* for each command separated by |, &&, ;, ,  
* check for input redirection if any (>, < and >>)
** update history in order of execution
* if | is there, do piping. Execute the left, store output in a out file, revert to stdout
* give the file as stdin to the right command. revert to stdin
* if && and/or ; is there, do the needful. 
* execute the && operations left to right while they exec succesfully
* execute ; seperated commands sequencially
*/

FILE* out; // used for input redirection and piping

typedef struct
{
    char* cmds[MAX_HISTORY];
    int size;
    int start;
} History;
History* h;
void init_history(void){
    h = (History*) malloc(sizeof(History));
    h->size = 0;
    h->start = 0;
    for (int i = 0; i < MAX_HISTORY; i++)
    {
        h->cmds[i] = NULL;
    }
}
void update_history(char* cmd){
    if(h->size < MAX_HISTORY){
        h->cmds[h->size++] = strdup(cmd);
    }
    else{
        int idx = h->start;
        free(h->cmds[idx]);
        h->cmds[idx] = strdup(cmd);
        h->start = (h->start+1)%MAX_HISTORY;
    }
}

void print_history(int n){
    if(h->size == 0) return;
    if(n < 0){
        n = h->size;
    }
    n = ((n < h->size)? n: h->size);
    int st_offset = h->size-n;

    for (int i = 0; i < n; i++)
    {
        int idx = (h->start+i+st_offset)%MAX_HISTORY;
        printf("%s\n", h->cmds[idx]);
    }
}

void free_history(void){
    if(!h) return;
    for (int i = 0; i < MAX_HISTORY; i++)
    {
        if(h->cmds[i]!=NULL) free(h->cmds[i]);
    }
    free(h);
}

void trim(char *s) {
    if (!s) return;
    char *start = s;
    while (*start && *start == ' ') start++;
    if (start != s) memmove(s, start, strlen(start) + 1);

    size_t n = strlen(s);
    while (n > 0 && s[n-1] == ' ') {
        s[n - 1] = '\0';
        n--;
    }
}

bool is_pipe(char* cmd){
    int l = 0;
    while(cmd[l]!='\0'){
        if(cmd[l] == '|') return 1;
        l++;
    }
    return 0;
}

int is_io_redir(char* cmd){
    int l = 0;
    while(cmd[l]!='\0'){
        if(cmd[l] == '>' && cmd[l+1] == '>') return 1;
        else if(cmd[l] == '>') return 2;
        else if(cmd[l] == '<') return 3;
        l++;
    }
    return 0;
}

int is_and(char* cmd){
    int l = 0;
    while(cmd[l+1]!='\0'){
        if(cmd[l] == '&' && cmd[l+1] == '&') return 1;
        else if(cmd[l] == '&') return -1;
        l++;
    }
    return 0;
}

size_t count_tokens(char** tokens) {
    size_t count = 0;
    if (!tokens) return 0;
    while (tokens[count] != NULL) {
        count++;
    }
    return count;
}

glob_t get_matches(char* pat, int* x){
    glob_t res;
    if(glob(pat, 0, NULL, &res)!=0){
        *x = 1;
    }
    return res;
}

bool is_wild(char* cmd){
    int l = 0;
    while(cmd[l]!='\0'){
        if(cmd[l++] == '*') return 1;
    }
    return 0;
}
int ct_wild(char** tokens){
    int ct = 0;
    for(int i = 0; tokens[i]; i++){
        if(is_wild(tokens[i])) ct++;
    }
    return ct;
}
void fill_wildcard(char*** p_tokens) {
    if (!p_tokens || !*p_tokens) {
        return;
    }
    int i = 0;
    while ((*p_tokens)[i] != NULL) {
        char* current_token = (*p_tokens)[i];
        if (strchr(current_token, '*') == NULL) {
            i++;
            continue;
        }
        int x = 0;
        glob_t glob_result = get_matches(current_token,&x);
        if(x == 1){
            i++;
            continue;
        }
        size_t n = glob_result.gl_pathc;
        if (n <= 1 && strcmp(current_token, glob_result.gl_pathv[0]) == 0) {
             globfree(&glob_result);
             i++;
             continue;
        }

        size_t old_ct = count_tokens(*p_tokens);
        size_t new_ct = old_ct - 1 + n;
        char** new_tokens = realloc(*p_tokens, (new_ct + 1) * sizeof(char*));
        *p_tokens = new_tokens;
        memmove(&(*p_tokens)[i + n], &(*p_tokens)[i + 1], (old_ct - i) * sizeof(char*));

        free(current_token);

        for (size_t j = 0; j < n; j++) {
            (*p_tokens)[i + j] = strdup(glob_result.gl_pathv[j]);
        }
        globfree(&glob_result);
        i += n;
    }
}

char **seq_tokenize(char * _cmd, int* argc){
    if (!_cmd) return NULL;
    char *cmd = strdup(_cmd);
    if (!cmd) return NULL;
    int ct = 1;
    for (char *p = cmd; *p; ++p) if (*p == ';') ++ct;

    char **tokens = malloc((ct + 1) * sizeof *tokens);
    if (!tokens) { free(cmd); return NULL; }

    int i = 0;
    char *start = cmd;
    for (char *p = cmd; ; ++p) {
        if (*p == ';' || *p == '\0') {
            size_t len = (size_t)(p - start);
            char *tok;
            if (len == 0) {
                tok = strdup("");
            } else {
                tok = malloc(len + 1);
                if (tok) {
                    memcpy(tok, start, len);
                    tok[len] = '\0';
                }
            }
            if (!tok) {
                for (int j = 0; j < i; ++j) free(tokens[j]);
                free(tokens);
                free(cmd);
                return NULL;
            }
            trim(tok);
            tokens[i++] = tok;
            if (*p == '\0') break;
            start = p + 1;
        }
    }
    tokens[i] = NULL;
    *argc = i;
    free(cmd);
    return tokens;
}
char **logic_tokenize(char * _cmd, int* argc){
    if (!_cmd) return NULL;

    char *cmd = strdup(_cmd);
    if (!cmd) return NULL;
    int ct = 1;
    for (size_t i = 0; cmd[i] && cmd[i+1]; ++i) {
        if (cmd[i] == '&' && cmd[i+1] == '&') ++ct;
    }

    char **tokens = malloc((ct + 1) * sizeof *tokens);
    if (!tokens) { free(cmd); return NULL; }

    int idx = 0;
    char *start = cmd;
    char *p = cmd;
    while (*p) {
        if (*p == '&' && p[1] == '&') {
            size_t len = (size_t)(p - start);
            char *tok;
            if (len == 0) {
                tok = strdup("");
            } else {
                tok = malloc(len + 1);
                if (tok) {
                    memcpy(tok, start, len);
                    tok[len] = '\0';
                }
            }

            if (!tok) {
                for (int j = 0; j < idx; ++j) free(tokens[j]);
                free(tokens);
                free(cmd);
                return NULL;
            }
            trim(tok);
            tokens[idx++] = tok;
            p += 2;
            start = p;
        } else {
            ++p;
        }
    }
    size_t len = strlen(start);
    char *tok = (len == 0) ? strdup("") : strdup(start);
    if (!tok) {
        for (int j = 0; j < idx; ++j) free(tokens[j]);
        free(tokens);
        free(cmd);
        return NULL;
    }
    trim(tok);
    tokens[idx++] = tok;
    tokens[idx] = NULL;
    *argc = idx;
    free(cmd);
    return tokens;
}

char** pipe_tokenize(char* _cmd, int* argc){
    if (!_cmd) return NULL;
    char *cmd = strdup(_cmd);
    if (!cmd) return NULL;
    int ct = 1;
    for (char *p = cmd; *p; ++p) if (*p == '|') ++ct;

    char **tokens = malloc((ct + 1) * sizeof *tokens);
    if (!tokens) { free(cmd); return NULL; }

    int idx = 0;
    char *start = cmd;
    for (char *p = cmd; ; ++p) {
        if (*p == '|' || *p == '\0') {
            size_t len = (size_t)(p - start);
            char *tok;
            if (len == 0) {
                tok = strdup("");
            } else {
                tok = malloc(len + 1);
                if (tok) {
                    memcpy(tok, start, len);
                    tok[len] = '\0';
                }
            }
            if (!tok) {
                for (int j = 0; j < idx; ++j) free(tokens[j]);
                free(tokens);
                free(cmd);
                return NULL;
            }
            trim(tok);
            tokens[idx++] = tok;

            if (*p == '\0') break;
            start = p + 1;
        }
    }

    tokens[idx] = NULL;
    *argc = idx;
    free(cmd);
    return tokens;
}

char** io_redir_tokenize(char* _cmd, int* argc){
    if (!_cmd) return NULL;

    char *cmd = strdup(_cmd);
    if (!cmd) return NULL;

    size_t len = strlen(cmd);
    int op_pos = -1;
    int op_len = 0; 

    for (size_t i = 0; i < len; ++i) {
        if (cmd[i] == '<' || cmd[i] == '>') {
            int this_len = (cmd[i] == '>' && i + 1 < len && cmd[i+1] == '>') ? 2 : 1;
            if (op_pos != -1) {
                free(cmd);
                return NULL;
            }
            op_pos = (int)i;
            op_len = this_len;
            if (this_len == 2) ++i;
        }
    }
    if (op_pos == -1) {
        free(cmd);
        return NULL;
    }
    char **tokens = malloc(3 * sizeof *tokens);
    if (!tokens) { free(cmd); return NULL; }
    size_t left_len = (size_t)op_pos;
    char *left = malloc(left_len + 1);
    if (!left) { free(tokens); free(cmd); return NULL; }
    if (left_len) memcpy(left, cmd, left_len);
    left[left_len] = '\0';
    trim(left);
    size_t right_start = (size_t)op_pos + (size_t)op_len;
    size_t right_len = (right_start <= len) ? (len - right_start) : 0;
    char *right = malloc(right_len + 1);
    if (!right) { free(left); free(tokens); free(cmd); return NULL; }
    if (right_len) memcpy(right, cmd + right_start, right_len);
    right[right_len] = '\0';
    trim(right);

    tokens[0] = left;
    tokens[1] = right;
    tokens[2] = NULL;
    *argc = 2;
    free(cmd);
    return tokens;
}

int remove_quotes(char *s) {
    if (!s) return 0;
    char *dst = s;
    int ct = 0;
    while (*s) {
        if (*s != '"' && *s != '\'') *dst++ = *s;
        else ct=(ct+1)%2;
        s++;
    }
    *dst = '\0';
    return ct;
}

char** split(char* _cmd, int* argc){
    if (!_cmd) return NULL;

    char *cmd = strdup(_cmd);
    if (!cmd) return NULL;
    int ct = 0;
    for (size_t i = 0; cmd[i]; ++i) {
        if (cmd[i] != ' ' && (i == 0 || cmd[i-1] == ' ')) ++ct;
    }
    if (ct == 0) {
        free(cmd);
        char **tokens = malloc(sizeof *tokens);
        if (!tokens) return NULL;
        tokens[0] = NULL;
        return tokens;
    }

    char **tokens = malloc((ct + 1) * sizeof *tokens);
    if (!tokens) { free(cmd); return NULL; }

    int idx = 0;
    char *tok = strtok(cmd, " ");
    while (tok) {
        trim(tok);
        char *copy = strdup(tok);
        if (!copy) {
            for (int j = 0; j < idx; ++j) free(tokens[j]);
            free(tokens);
            free(cmd);
            return NULL;
        }
        remove_quotes(copy);
        tokens[idx++] = copy;
        tok = strtok(NULL, " ");
    }
    tokens[idx] = NULL;
    *argc = idx;
    free(cmd);
    return tokens;
}

int to_int(char* s, int* x){
    int ans = atoi(s);
    int l = 0;
    while(s[l++] == '0') continue;
    if(ans || (ans == 0 && s[l] == '\0')) *x = 1;
    else{
        *x = 0;
    }
    return ans;
}

void clear(char** list){
    int l = 0;
    while(list[l]){
        free(list[l++]);
    }
    free(list);
}

int run(char** tokens){
    if (!tokens || tokens[0] == NULL) return 1;

    
    if (strcmp(tokens[0], "cd") == 0) {
        char *path = NULL;

        if (tokens[1] == NULL || strcmp(tokens[1], "~") == 0) {
            if (tokens[1] != NULL && tokens[2] != NULL) { 
                 ERROR;
                 return 1;
            }
            path = getenv("HOME");
            if (path == NULL) {
                fprintf(stderr, "cd: HOME not set\n");
                return 1;
            }
        } else if (strcmp(tokens[1], "-") == 0) {
            if (tokens[2] != NULL) {
                ERROR;
                return 1;
            }
            path = getenv("OLDPWD");
            if (path == NULL) {
                fprintf(stderr, "cd: OLDPWD not set\n");
                return 1;
            }
        } else {
            if (tokens[2] != NULL) {
                ERROR;
                return 1;
            }
            path = tokens[1];
        }

        char* old_pwd = getcwd(NULL, 0);
        if (!old_pwd) {
            perror("getcwd");
            return 1;
        }

        if (chdir(path) != 0) {
            perror(path);
            free(old_pwd);
            return 1;
        }

        setenv("OLDPWD", old_pwd, 1);
        free(old_pwd);
        char* new_pwd = getcwd(NULL, 0);
        if (new_pwd) {
            setenv("PWD", new_pwd, 1);
            free(new_pwd);
        }
        return 0;
    }
    else if (strcmp("history", tokens[0]) == 0){
        if (tokens[1] == NULL) {
            print_history(-1);
            return 0;
        }
        if (tokens[2] != NULL) {
            ERROR;
            return 1;
        }

        int ok;
        int a = to_int(tokens[1], &ok);
        if (!ok) {
            ERROR;
            return 1;
        }
        print_history(a);
        return 0;
    }
    else {
        pid_t pid = fork();
        int status;

        if (pid == -1) {
            perror("fork failed");
            return 1;
        }
        else if (pid == 0) {
            execvp(tokens[0], tokens);
            ERROR;
            _exit(127);
        }
        else {
            if (waitpid(pid, &status, 0) == -1) {
                perror("waitpid");
                return 1;
            }
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return 0;
            return 1;
        }
    }
    return 1;
}

int run_no_fork(char** tokens){
    if (!tokens || tokens[0] == NULL) return 1;

    if (strcmp("cd", tokens[0]) == 0){
        if (tokens[1] == NULL) {
            ERROR;
            return 1;
        }
        if (tokens[2] != NULL) {
            ERROR;
            return 1;
        }
        if (chdir(tokens[1]) == 0) {
            return 0;
        } else {
            ERROR;
            return 1;
        }
    }
    else if (strcmp("history", tokens[0]) == 0){
        if (tokens[1] == NULL) {
            print_history(-1);
            return 0;
        }
        if (tokens[2] != NULL) {
            ERROR;
            return 1;
        }

        int ok;
        int a = to_int(tokens[1], &ok);
        if (!ok) {
            ERROR;
            return 1;
        }
        print_history(a);
        return 0;
    }
    else {
        if(execvp(tokens[0], tokens) == -1){
            ERROR;
            return 1;
        }
    }
    return 1;
}

int run_io_redir(char* cmd, int mode){
    int argc = 0, largc = 0, rargc = 0;
    char** io_tokens = io_redir_tokenize(cmd, &argc);
    if(argc == 0){
        ERROR;
        return 1;
    }

    char** ltokens = split(io_tokens[0], &largc); 
    char** rtokens = split(io_tokens[1], &rargc);
    
    if (!ltokens || !rtokens){
        clear(io_tokens);
        clear(ltokens);
        clear(rtokens);
        return 1;
    }
    fill_wildcard(&ltokens);
    
    if (rtokens[0] == NULL || rtokens[0][0] == '\0'){
        ERROR;
        clear(io_tokens);
        clear(ltokens);
        clear(rtokens);
        return 1;
    }
    char *file_name = rtokens[0];
    
    int ret = 0;
    
    if (mode == 1 || mode == 2) {
        int saved_stdout = dup(STDOUT_FILENO);
        if (saved_stdout < 0){
            perror("dup");
            ret = 1;
            goto cleanup;
        }

        int flags = O_WRONLY | O_CREAT | (mode == 1 ? O_APPEND : O_TRUNC);
        int newout = open(file_name, flags, 0666);
        if (newout < 0){
            perror("open");
            close(saved_stdout);
            ret = 1;
            goto cleanup;
        }

        if (dup2(newout, STDOUT_FILENO) < 0){
            perror("dup2");
            close(newout);
            close(saved_stdout);
            ret = 1;
            goto cleanup;
        }
        if (ltokens[0]) {
            if (run(ltokens) != 0) {
                ret = 1;
            }
        }
        if (dup2(saved_stdout, STDOUT_FILENO) < 0) perror("dup2-restore");
        close(newout);
        close(saved_stdout);
        int j = 1;
        while (rtokens[j]) {
            printf("%s ", rtokens[j++]);
        }
        if (j > 1) putchar('\n');
    }
    else if (mode == 3) {

        if (ltokens[0] == NULL) {
            pid_t pid = fork();
            if (pid < 0){
                perror("fork");
                ret = 1;
                goto cleanup;
            } else if (pid == 0) {
                char *argv[] = {"cat", file_name, NULL};
                execvp("cat", argv);
                perror("execvp");
                _exit(127);
            } else {
                int status;
                if (waitpid(pid, &status, 0) == -1) {
                    perror("waitpid");
                    ret = 1;
                } else if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                    ret = 1;
                }
            }
        } else {
            int newin = open(file_name, O_RDONLY);
            if (newin < 0){
                perror("open");
                ret = 1;
                goto cleanup;
            }

            int saved_stdin = dup(STDIN_FILENO);
            if (saved_stdin < 0){
                perror("dup");
                close(newin);
                ret = 1;
                goto cleanup;
            }

            if (dup2(newin, STDIN_FILENO) < 0){
                perror("dup2");
                close(newin);
                close(saved_stdin);
                ret = 1;
                goto cleanup;
            }

            close(newin);

            if (run(ltokens) != 0) {
                ret = 1;
            }

            if (dup2(saved_stdin, STDIN_FILENO) < 0) perror("dup2-restore");
            close(saved_stdin);
        }
    }
    else {
        ERROR;
        ret = 1;
    }

cleanup:
    clear(io_tokens);
    clear(ltokens);
    clear(rtokens);
    return ret;
}

static int wait_for(pid_t pid) {
    int status;
    while (1) {
        if (waitpid(pid, &status, 0) < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (WIFEXITED(status)) return WEXITSTATUS(status);
        if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
        return -1;
    }
}

int run_pipe(char* cmd) {
    int argc = 0, largc = 0, rargc = 0;
    char** pipe_tokens = pipe_tokenize(cmd, &argc);
    if (!pipe_tokens || !pipe_tokens[0] || !pipe_tokens[1]) {
        if (pipe_tokens) clear(pipe_tokens);
        ERROR;
        return 1;
    }
    char** ltokens = split(pipe_tokens[0], &largc);
    char** rtokens = split(pipe_tokens[1], &rargc);

    if (!ltokens || !rtokens || !ltokens[0] || !rtokens[0]) {
        ERROR;
        clear(pipe_tokens);
        if (ltokens) clear(ltokens);
        if (rtokens) clear(rtokens);
        return 1;
    }

    fill_wildcard(&ltokens);
    fill_wildcard(&rtokens);

    int pipe_fd[2];
    if (pipe(pipe_fd) == -1) {
        perror("pipe");
        clear(pipe_tokens); clear(ltokens); clear(rtokens);
        return 1;
    }

    pid_t pid1 = fork();

    if (pid1 == 0) {
        close(pipe_fd[0]);
        dup2(pipe_fd[1], STDOUT_FILENO);
        close(pipe_fd[1]);

        run_no_fork(ltokens);
        _exit(1);
    }

    pid_t pid2 = fork();

    if (pid2 == 0) {
        close(pipe_fd[1]);
        dup2(pipe_fd[0], STDIN_FILENO);
        close(pipe_fd[0]);

        run_no_fork(rtokens);
        _exit(1);
    }

    close(pipe_fd[0]);
    close(pipe_fd[1]);

    int status1 = wait_for(pid1);
    int status2 = wait_for(pid2);

    clear(pipe_tokens);
    clear(ltokens);
    clear(rtokens);

    if (status1 != 0 || status2 != 0) return 1;
    return 0;
}

char* get_match(char* pat){
    int x = 0;
    glob_t m = get_matches(pat,&x);
    if(x == 1 || m.gl_pathc != 1){
        globfree(&m);
        return NULL; 
    }
    else{
        char* match = strdup(m.gl_pathv[0]);
        globfree(&m);
        return match;
    }
}

struct termios og_termios;

void disable_raw_mode(void){
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &og_termios);
}

void enable_raw_mode(void){
    tcgetattr(STDIN_FILENO, &og_termios);
    atexit(disable_raw_mode);
    struct termios raw = og_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

int main(void)

{
    init_history();
    char* cmd = malloc(2049*sizeof(char));
    bool should_exit = false;

    while (1) {
        printf("MTL458 > ");
        fflush(stdout);

        enable_raw_mode();

        char c;
        int i = 0;
        while(read(STDIN_FILENO, &c, 1) == 1){
            bool f = 0;
            if(c == '\t'){
                int j = i;
                    while(j && !isspace(cmd[j-1])){
                        j--;
                    }
                    char pat[i-j+2];
                    strncpy(pat,&cmd[j],i-j);
                    pat[i-j] = '*';
                    pat[i-j+1] = '\0';
                    char* match = get_match(pat);
                    if(match){
                        strcpy(&cmd[i],&match[i-j]);
                        printf("%s", &match[i-j]);
                        fflush(stdout);
                        i = strlen(cmd);
                    }
            }
            else switch(c){
                case '\r':
                case '\n':
                    printf("\n");
                    f = 1;
                    break;
                case 127:
                    if(i > 0){
                        cmd[--i] = '\0';
                        printf("\b \b");
                        fflush(stdout);
                    }
                    break;
                default:
                    if (i < 2048 && isprint(c)) {
                        cmd[i++] = c;
                        cmd[i] = '\0';
                        printf("%c", c);
                        fflush(stdout);
                    }
                    break;
            }
            if(f) break;
        }
        if(i == 0) continue;;
        int seq_argc = 0;
        char **seq_tokens = seq_tokenize(cmd,&seq_argc);
        if (!seq_tokens || seq_argc == 0) continue;
        update_history(cmd);

        for (int i = 0; seq_tokens[i]; ++i) {
            if (seq_tokens[i][0] == '\0') continue;

            int and_val = is_and(seq_tokens[i]);
            if (and_val == 1) {
                int logic_argc = 0;
                char **logic_tokens = logic_tokenize(seq_tokens[i],&logic_argc);
                if (!logic_tokens || logic_argc == 0) { ERROR; continue; }

                bool ok = true;
                for (int j = 0; logic_tokens[j]; ++j) {
                    if (logic_tokens[j][0] == '\0') { ERROR; ok = false; break; }
                }

                if (ok) {
                    for (int j = 0; logic_tokens[j] && !should_exit; ++j) {
                        int mode = is_pipe(logic_tokens[j]);
                        if (mode) {
                            if (run_pipe(logic_tokens[j]) == 1) break;
                        } else {
                            mode = is_io_redir(logic_tokens[j]);
                            if (mode) {
                                if (run_io_redir(logic_tokens[j], mode) == 1) break;
                            } else {
                                int argc = 0;
                                char **tokens = split(logic_tokens[j],&argc);
                                if (!tokens || argc == 0) { ERROR; break; }
                                fill_wildcard(&tokens);
                                if (tokens[0] && strcmp(tokens[0], "exit") == 0) {
                                    clear(tokens);
                                    should_exit = true;
                                    break;
                                }
                                if (run(tokens) == 1) {
                                    clear(tokens);
                                    break;
                                }
                                clear(tokens);
                            }
                        }
                    }
                }
                clear(logic_tokens);
                if (should_exit) break;
            }
            else if (and_val == -1) {
                ERROR;
            }
            else {
                int mode = is_pipe(seq_tokens[i]);
                if (mode) {
                    run_pipe(seq_tokens[i]);
                } else {
                    mode = is_io_redir(seq_tokens[i]);
                    if (mode) {
                        run_io_redir(seq_tokens[i], mode);
                    } else {
                        int argc = 0;
                        char **tokens = split(seq_tokens[i], &argc);
                        if (tokens && argc) {
                            fill_wildcard(&tokens);
                            if (strcmp(tokens[0], "exit") == 0) {
                                clear(tokens);
                                should_exit = true;
                            } else {
                                run(tokens);
                                clear(tokens);
                            }
                        } else {
                            clear(tokens);
                        }
                    }
                }
                if (should_exit) break;
            }
        }

        clear(seq_tokens);

        if (should_exit) break;
    } 

    free(cmd);
    free_history();

    return 0;
}