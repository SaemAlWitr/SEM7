// myshell.c - minimal shell using fork + execvp
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>

#define PROMPT "\nmysh$ "
// simple tokenizer (splits on whitespace). Not handling quotes/escapes.
char **tokenize(char *line, int *argc_out) {
    int capacity = 8;
    char **argv = malloc(sizeof(char*) * capacity);
    int argc = 0;
    char *tok = strtok(line, " \t\r\n");
    while (tok) {
        if (argc + 1 >= capacity) {
            capacity *= 2;
            argv = realloc(argv, sizeof(char*) * capacity);
        }
        argv[argc++] = tok;
        tok = strtok(NULL, " \t\r\n");
    }
    argv[argc] = NULL;
    *argc_out = argc;
    return argv;
}

int main(void) {
    // Avoid zombie children for background jobs: reap automatically.
    // int append = 0;
    // int flags = O_WRONLY | O_CREAT | (append ? O_APPEND : O_TRUNC);
    // int fd = open("file4.txt", flags, 0666);
    // close(fd);
    // FILE* f = fopen("file4.txt","w");
    // fclose(f);
    char *line = NULL;
    size_t n = 0;

    while (1) {
        // print prompt
        int status;
        pid_t w;
        while ((w = waitpid(-1, &status, WNOHANG)) > 0) {
            // optional: report reaped background child
            printf("[reaped %d]\n", w);
        }
        if (isatty(STDIN_FILENO)) {
            fputs(PROMPT, stdout);
            fflush(stdout);
        }

        ssize_t len = getline(&line, &n, stdin);
        if (len < 0) break; // EOF or error (Ctrl-D)

        // strip newline handled by tokenizer
        int argc;
        char *line_copy = strdup(line); // tokenize modifies buffer
        char **argv = tokenize(line_copy, &argc);
        if (argc == 0) { free(argv); free(line_copy); continue; }

        // handle builtins
        if (strcmp(argv[0], "exit") == 0) {
            free(argv); free(line_copy); break;
        }
        if (strcmp(argv[0], "cd") == 0) {
            const char *dir = (argc > 1) ? argv[1] : getenv("HOME");
            if (chdir(dir) != 0) perror("Invalid Command");
            free(argv); free(line_copy); continue;
        }

        // check for background job (trailing &)
        int background = 0;
        if (argc > 0 && strcmp(argv[argc-1], "&") == 0) {
            background = 1;
            argv[argc-1] = NULL; // remove &
        }

        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            free(argv); free(line_copy);
            continue;
        } else if (pid == 0) {
            // child: execute command (search PATH)
            execvp(argv[0], argv);
            // if execvp returns, there was an error
            perror("execvp");
            _exit(127);
        } else {
            // parent
            if (!background) {
                int status;
                if (waitpid(pid, &status, 0) < 0) perror("waitpid");
            } else {
                // for background, we won't wait here. Child is reaped by SIGCHLD=SIG_IGN.
                printf("[bg] pid %d\n", pid);
            }
        }

        free(argv);
        free(line_copy);
    }

    free(line);
    puts("Bye.");
    return 0;
}
