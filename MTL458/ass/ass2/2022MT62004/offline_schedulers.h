#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <sys/time.h>
#include <stdbool.h>
#include <fcntl.h>
#include <string.h>

typedef struct {
    char *command;
    bool finished;
    bool error;
    uint64_t start_time;
    uint64_t completion_time;
    uint64_t turnaround_time;
    uint64_t waiting_time;
    uint64_t response_time;
    uint64_t burst_time;
    bool started;
    int process_id;
} Process;

uint64_t get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)(tv.tv_sec * 1000 + tv.tv_usec / 1000);
}

void parse_command(char *command, char **argv) {
    int i = 0;
    char *token = strtok(command, " ");
    while (token != NULL && i < 99) {
        argv[i++] = token;
        token = strtok(NULL, " ");
    }
    argv[i] = NULL;
}

void write_to_csv(const char *filename, Process p[], int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return;
    
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%s,%s,%s,%lu ,%lu ,%lu ,%lu \n",
                p[i].command,
                p[i].finished ? "Yes" : "No",
                p[i].error ? "Yes" : "No",
                p[i].completion_time,
                p[i].turnaround_time,
                p[i].waiting_time,
                p[i].response_time);
    }
    
    fclose(fp);
}

void FCFS(Process p[], int n) {
    uint64_t program_start = get_time_ms();
    
    for (int i = 0; i < n; i++) {
        p[i].finished = false;
        p[i].error = false;
        p[i].started = false;
        p[i].start_time = 0;
    }
    
    for (int i = 0; i < n; i++) {
        // fork exec the process
        pid_t pid = fork();
        if (pid == 0) {
            char *argv[100];
            char cmd_copy[1001];
            strcpy(cmd_copy, p[i].command);
            parse_command(cmd_copy, argv);
            execvp(argv[0], argv);
            exit(1);
        } else if (pid > 0) {
            p[i].process_id = pid;
            p[i].started = true;
            p[i].start_time = get_time_ms();
            
            p[i].response_time = p[i].start_time - program_start;
            p[i].waiting_time = p[i].response_time;
            
            printf("%s, %lu, ", p[i].command, p[i].start_time - program_start);
            
            int status;
            waitpid(pid, &status, 0);
            
            p[i].completion_time = get_time_ms() - program_start;
            printf("%lu\n", p[i].completion_time);
            
            if (WIFEXITED(status)) {
                int exit_status = WEXITSTATUS(status);
                if (exit_status == 0) {
                    p[i].finished = true;
                    p[i].error = false;
                } else {
                    p[i].finished = false;
                    p[i].error = true;
                }
            } else {
                p[i].finished = false;
                p[i].error = true;
            }
            p[i].turnaround_time = p[i].completion_time;
        }
    }
    
    write_to_csv("result_offline_FCFS.csv", p, n);
}

void RoundRobin(Process p[], int n, int quantum_ms) {
    uint64_t program_start = get_time_ms();
    
    for (int i = 0; i < n; i++) {
        p[i].finished = false;
        p[i].error = false;
        p[i].started = false;
        p[i].start_time = 0;
        p[i].process_id = -1;
    }
    
    int remaining = n;
    
    while (remaining > 0) {
        for (int i = 0; i < n; i++) {
            if (p[i].finished || p[i].error) continue;
            
            if (!p[i].started) {
                pid_t pid = fork();
                
                if (pid == 0) {
                    char *argv[100];
                    char cmd_copy[1001];
                    strcpy(cmd_copy, p[i].command);
                    parse_command(cmd_copy, argv);
                    execvp(argv[0], argv);
                    exit(1);
                } else if (pid > 0) {
                    p[i].process_id = pid;
                    p[i].started = true;
                    p[i].start_time = get_time_ms();
                    p[i].response_time = p[i].start_time - program_start;
                    p[i].burst_time = 0;
                    p[i].waiting_time = p[i].response_time;
                }
            } else {
                kill(p[i].process_id, SIGCONT);
            }
            
            uint64_t slice_start = get_time_ms();
            printf("%s, %lu, ", p[i].command, slice_start - program_start);
            
            uint64_t elapsed = 0;
            int status;
            if(remaining > 1){
                while (elapsed < quantum_ms) {
                    pid_t result = waitpid(p[i].process_id, &status, WNOHANG);
                    
                    if (result == p[i].process_id) {
                        p[i].completion_time = get_time_ms() - program_start;
                        printf("%lu\n", p[i].completion_time);
                        
                        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                            p[i].finished = true;
                        } else {
                            p[i].error = true;
                            break;
                        }
                        p[i].turnaround_time = p[i].completion_time;
                        p[i].burst_time += get_time_ms()-slice_start;
                        p[i].waiting_time = p[i].turnaround_time-p[i].burst_time; 
                        remaining--;
                        break;
                    }
                    
                    // usleep(10000);
                    elapsed = get_time_ms() - slice_start;
                }
                
                if (!p[i].finished && !p[i].error) {
                    kill(p[i].process_id, SIGSTOP);
                    uint64_t slice_end = get_time_ms();
                    p[i].burst_time+=slice_end-slice_start;
                    printf("%lu\n", slice_end - program_start);
                }
            }
            else{
                pid_t result = waitpid(p[i].process_id, &status, 0);
                    
                if (result == p[i].process_id) {
                    p[i].completion_time = get_time_ms() - program_start;
                    printf("%lu\n", p[i].completion_time);
                    
                    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                        p[i].finished = true;
                    } else {
                        p[i].error = true;
                    }
                    p[i].turnaround_time = p[i].completion_time;
                    p[i].burst_time += get_time_ms()-slice_start;
                    p[i].waiting_time = p[i].turnaround_time-p[i].burst_time; 
                    remaining--;
                }
            }

        }
    }
    
    write_to_csv("result_offline_RR.csv", p, n);
}

void MultiLevelFeedbackQueue(Process p[], int n, int quantum2, int quantum1, int quantum0, int boostTime) {
    uint64_t program_start = get_time_ms();
    uint64_t last_boost = 0;
    int quantums[3] = {quantum0, quantum1, quantum2};
    int queue[n];
    
    for (int i = 0; i < n; i++) {
        p[i].finished = false;
        p[i].error = false;
        p[i].started = false;
        p[i].start_time = 0;
        p[i].process_id = -1;
        queue[i] = 2;
    }
    
    int remaining = n;
    
    while (remaining > 0) {
        uint64_t current_time = get_time_ms() - program_start;
        if (boostTime > 0 && (current_time - last_boost) >= boostTime) {
            for (int i = 0; i < n; i++) {
                if (!p[i].finished && !p[i].error) {
                    queue[i] = 2;
                }
            }
            last_boost = current_time;
        }
        
        int selected = -1;
        for (int q = 2; q >= 0; q--) {
            for (int i = 0; i < n; i++) {
                if (!p[i].finished && !p[i].error && queue[i] == q) {
                    selected = i;
                    break;
                }
            }
            if (selected != -1) break;
        }
        
        if (selected == -1) break;
        
        int i = selected;
        int current_quantum = quantums[queue[i]];
        
        if (!p[i].started) {
            pid_t pid = fork();
            
            if (pid == 0) {
                char *argv[100];
                char cmd_copy[1001];
                strcpy(cmd_copy, p[i].command);
                parse_command(cmd_copy, argv);
                execvp(argv[0], argv);
                exit(1);
            } else if (pid > 0) {
                p[i].process_id = pid;
                p[i].started = true;
                p[i].start_time = get_time_ms();
                p[i].response_time = p[i].start_time - program_start;
                p[i].burst_time = 0;
            }
        } else {
            kill(p[i].process_id, SIGCONT);
        }
        
        uint64_t slice_start = get_time_ms();
        printf("%s, %lu, ", p[i].command, slice_start - program_start);
        
        uint64_t elapsed = 0;
        int status;
        bool completed = false;
        
        while (elapsed < current_quantum) {
            pid_t result = waitpid(p[i].process_id, &status, WNOHANG);
            
            if (result == p[i].process_id) {
                p[i].completion_time = get_time_ms() - program_start;
                printf("%lu\n", p[i].completion_time);
                
                if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                    p[i].finished = true;
                } else {
                    p[i].error = true;
                }
                p[i].turnaround_time = p[i].completion_time;
                p[i].burst_time += get_time_ms()-slice_start;
                p[i].waiting_time = p[i].turnaround_time-p[i].burst_time;
                remaining--;
                completed = true;
                break;
            }
            
            // usleep(10000);
            elapsed = get_time_ms() - slice_start;
        }
        
        if (!completed) {
            kill(p[i].process_id, SIGSTOP);
            uint64_t slice_end = get_time_ms();
            printf("%lu\n", slice_end - program_start);
            p[i].burst_time+=slice_end-slice_start;
            if (queue[i] > 0) {
                queue[i]--;
            }
        }
    }
    
    write_to_csv("result_offline_MLFQ.csv", p, n);
}