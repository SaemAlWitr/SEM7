#pragma once

//Can include any other headers as needed
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <sys/time.h>
#include <stdbool.h>
#include <string.h>
#include <fcntl.h>

char input_buffer[1001];

typedef struct {
    char *command;
    bool finished;
    bool error;    
    uint64_t start_time;
    uint64_t completion_time;
    uint64_t turnaround_time;
    uint64_t waiting_time;
    uint64_t response_time;
    uint64_t arrival_time;
    uint64_t actual_burst_time;
    bool started; 
    int process_id;
    int burst_time;

} Process;

typedef struct {
    char command[1001];
    int exec_count;
    uint64_t burst_times[50];
} CommandHistory;


CommandHistory history[50];
int history_count = 0;

uint64_t get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)(tv.tv_sec * 1000 + tv.tv_usec / 1000);
}

bool stdin_has_data() {
    struct timeval tv = {0, 0};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    return select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0;
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

void write_to_csv(const char *filename, Process* p) {
    FILE *fp = fopen(filename, "a");
    if (!fp) return;
    fprintf(fp, "%s,%s,%s,%lu,%lu,%lu,%lu\n",
            p->command,
            p->finished ? "Yes" : "No",
            p->error ? "Yes" : "No",
            p->completion_time,
            p->turnaround_time,
            p->waiting_time,
            p->response_time);

    fclose(fp);
}

int read_new_commands(Process p[], int _n, uint64_t program_start) {
    int n = _n;
    
    while (stdin_has_data() && fgets(input_buffer, sizeof(input_buffer), stdin) != NULL) {
        // Remove \n
        input_buffer[strcspn(input_buffer, "\n")] = 0;
        
        if (strlen(input_buffer) > 0 && n < 50) {
            p[n].command = strdup(input_buffer);
            p[n].finished = false;
            p[n].error = false;
            p[n].started = false;
            p[n].process_id = -1;
            p[n].arrival_time = get_time_ms() - program_start;
            p[n].actual_burst_time = 0;
            n++;
        }
    }
    
    return n;
}

uint64_t get_average_burst_time(const char *command, int k) {
    for (int i = 0; i < history_count; i++) {
        if (strcmp(history[i].command, command) == 0) {
            if (history[i].exec_count == 0) {
                return 1000;
            }
            
            int count = history[i].exec_count < k ? history[i].exec_count : k;
            uint64_t sum = 0;
            int start = history[i].exec_count - count;
            
            for (int j = start; j < history[i].exec_count; j++) {
                sum += history[i].burst_times[j];
            }
            
            return sum / count;
        }
    }
    
    return 1000;
}

uint64_t get_average_burst_time_mlfq(const char *command, int k) {
    for (int i = 0; i < history_count; i++) {
        if (strcmp(history[i].command, command) == 0) {
            if (history[i].exec_count == 0) {
                return -1;
            }
            
            int count = history[i].exec_count < k ? history[i].exec_count : k;
            uint64_t sum = 0;
            int start = history[i].exec_count - count;
            
            for (int j = start; j < history[i].exec_count; j++) {
                sum += history[i].burst_times[j];
            }
            
            return sum / count;
        }
    }
    
    return -1;
}

void update_history(const char *command, uint64_t burst_time) {
    for (int i = 0; i < history_count; i++) {
        if (strcmp(history[i].command, command) == 0) {
            if (history[i].exec_count < 50) {
                history[i].burst_times[history[i].exec_count++] = burst_time;
            }
            return;
        }
    }
    
    strcpy(history[history_count].command, command);
    history[history_count].burst_times[0] = burst_time;
    history[history_count].exec_count = 1;
    history_count++;
}


// Function prototypes
void ShortestJobFirst(int k){ // k is the last-k executions whose avg has to be taken
    FILE* f = fopen("result_online_SJF.csv", "w");
    fclose(f);
    uint64_t program_start = get_time_ms();
    Process p[50];
    
    int n = 0;
    n = read_new_commands(p, n, program_start);
    while(1){
        // find process with shortest b time
        int sel = -1;
        uint64_t min_burst = UINT64_MAX;
        for (int i = 0; i < n; i++)
        {
            if(!p[i].finished && !p[i].error){
                uint64_t avg_burst = get_average_burst_time(p[i].command, k);
                if(min_burst > avg_burst){
                    min_burst=avg_burst;
                    sel = i;
                }
            }
        }
        if(sel != -1){

            Process* process = &p[sel];

            pid_t pid = fork();
            if(pid == 0){
                char* argv[100];
                char cmd_cpy[1001];
                strcpy(cmd_cpy, process->command);
                parse_command(cmd_cpy, argv);
                execvp(argv[0], argv);
                exit(1);
            }
            else if(pid > 0){
                process->process_id = pid;
                process->start_time = get_time_ms()-program_start;
                process->response_time = process->start_time-process->arrival_time;
            }
            uint64_t exec_start = get_time_ms();
            printf("%s, %lu, ", process->command, exec_start - program_start);
            
            // Wait for process to complete
            int status;
            waitpid(process->process_id, &status, 0);
            
            uint64_t exec_end = get_time_ms();
            process->actual_burst_time += (exec_end - exec_start);
            process->completion_time = exec_end - program_start;
            
            printf("%lu\n", process->completion_time);
            
            // Check exit status
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                process->finished = true;
                process->error = false;
                update_history(process->command, process->actual_burst_time);
            } else {
                process->finished = false;
                process->error = true;
            }
            
            process->turnaround_time = process->completion_time - process->arrival_time;
            process->waiting_time = process->turnaround_time - process->actual_burst_time;
            write_to_csv("result_online_SJF.csv",process);
        }
        // Check for new commands
        n = read_new_commands(p, n, program_start);
    }

}
void MultiLevelFeedbackQueue(int quantum2, int quantum1, int quantum0, int boostTime){
    FILE* f = fopen("result_online_MLFQ.csv", "w");
    fclose(f);
    Process p[50];
    int n = 0;
    uint64_t program_start = get_time_ms();
    n = read_new_commands(p,n,program_start);
    
    int queue[50];
    int quantums[3] = {quantum0, quantum1, quantum2}; // quantum0 is the highest p queue
    uint64_t last_boost = program_start;
    for (int i = 0; i < n; i++) {
        queue[i] = 1;
    }
    while (1){
        uint64_t current_time = get_time_ms();
        if((current_time-last_boost) >= boostTime){
            // printf("%lu,%lu,booting\n", current_time,last_boost);
            for (int i = 0; i < n; i++)
            {
                if(!p[i].finished && !p[i].error) queue[i] = 2;
            }
            last_boost = current_time;
        }
        int n_ = n;
        n = read_new_commands(p, n, program_start);
        for (int i = n_; i < n; i++)
        {
            uint64_t avg_bst = get_average_burst_time_mlfq(p[i].command, 50); // putting lagest posble k
            if(avg_bst == -1) queue[i]=1;
            else if(avg_bst < quantums[2])queue[i] = 2;
            else if(avg_bst < quantums[1]) queue[i] = 1;
            else queue[i] = 0;
        }
        int sel = -1;
        for (int q = 2; q >= 0; q--) {
            for (int i = 0; i < n; i++) {
                if (!p[i].finished && !p[i].error && queue[i] == q) {
                    sel = i;
                    break;
                }
            }
            if (sel != -1) break;
        }
        if(sel!=-1){
            // printf("%d, %d\n",sel, queue[sel]);
            Process* process = &p[sel];
            int curr_quantum = quantums[queue[sel]];
            if(!process->started){
                pid_t pid = fork();
                if(pid == 0){
                    char* argv[100];
                    char cmd_cpy[1001];
                    strcpy(cmd_cpy, process->command);
                    parse_command(cmd_cpy, argv);
                    execvp(argv[0], argv);
                    exit(1);
                }
                else if(pid > 0){
                    process->process_id = pid;
                    process->started=true;
                    process->start_time = get_time_ms()-program_start;
                    process->response_time = process->start_time-process->arrival_time;
                }
            }
            else kill(process->process_id, SIGCONT);
            uint64_t slice_start = get_time_ms();
            printf("%s, %lu, ", process->command, slice_start - program_start); 
            // wait for process to finish/ the quantum to end
            uint64_t elapsed = 0;
            int status;
            bool completed = false;
            while(elapsed < curr_quantum){
                pid_t res = waitpid(process->process_id, &status, WNOHANG);
                if(res == process->process_id){
                    uint64_t exec_end = get_time_ms();
                    process->actual_burst_time += (exec_end - slice_start);
                    process->completion_time = exec_end - program_start;
                    
                    printf("%lu\n", process->completion_time);
                    
                    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                        process->finished = true;
                        process->error = false;
                        update_history(process->command, process->actual_burst_time);
                    } else {
                        process->error = true;
                    }
                    
                    process->turnaround_time = process->completion_time - process->arrival_time;
                    process->waiting_time = process->turnaround_time - process->actual_burst_time;
                    completed = true;
                    write_to_csv("result_online_MLFQ.csv",process);
                    break;
                }
                // usleep(2000); // wait 2ms before checking again
                elapsed = get_time_ms()- slice_start;
            }
            if(!completed){
                // printf("%d, ",n);
                kill(process->process_id, SIGSTOP);
                uint64_t slice_end = get_time_ms();
                process->actual_burst_time+=slice_end-slice_start;
                printf("%lu\n", slice_end - program_start);
                if(queue[sel] > 0) queue[sel]--;
            }
        }


        
    }
    

}
