#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <stdbool.h>
// Include your Headers below
#define MAGIC_NUMBER 1000000007
#define BLOCK_SIZE 4096
#define FIRST_FIT 1
#define NEXT_FIT 2
#define BEST_FIT 3
#define WORST_FIT 4
#define BUDDY 5
// init the heap. Use mmap to allocate 4096 bytes of memory.

int mode = 0;

typedef struct __node_t {
    size_t size;
    struct __node_t *next;
    struct __node_t *prev;
    } node_t;

static node_t* head = (node_t*)MAP_FAILED;

typedef struct __header_t {
    size_t size;
    size_t magic;
 } header_t;

// You are not allowed to use the function malloc and calloc directly .

static void init_heap() {
    head = (node_t*)mmap(NULL, BLOCK_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if(head == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    head->size = BLOCK_SIZE-sizeof(node_t);
    head->next = NULL;
    head->prev=NULL;
}

typedef struct avl_node {
    size_t size;
    node_t *blk;
    struct avl_node *left;
    struct avl_node *right;
    int height;
} avl_node_t;
static avl_node_t *avl_root = NULL;

avl_node_t* avl_node_new(node_t *blk) {
    avl_node_t *n = (avl_node_t*)mmap(NULL, sizeof(avl_node_t), PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (n == MAP_FAILED) return NULL;
    n->size = blk->size;
    n->blk = blk;
    n->left = n->right = NULL;
    n->height = 1;
    return n;
}

static int height_of(avl_node_t *n) { return (n ? n->height : 0);}
static int max(int a, int b) { return a > b ? a : b; }
static void update_height(avl_node_t *n) { if (n) n->height = 1 + max(height_of(n->left), height_of(n->right)); }
static int key_cmp(size_t asz, size_t bsz, uintptr_t a, uintptr_t b) {
    if (asz < bsz) return -1;
    if (asz > bsz) return 1;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static avl_node_t* rotate_right(avl_node_t *y) {
    avl_node_t *x = y->left;
    avl_node_t *t2 = x->right;
    x->right = y;
    y->left = t2;
    update_height(y);
    update_height(x);
    return x;
}
static avl_node_t* rotate_left(avl_node_t *x) {
    avl_node_t *y = x->right;
    avl_node_t *t2 = y->left;
    y->left = x;
    x->right = t2;
    update_height(x);
    update_height(y);
    return y;
}
static int get_balance(avl_node_t *n) {
    if (!n) return 0;
    return height_of(n->left) - height_of(n->right);
}

static avl_node_t* avl_insert(avl_node_t *root, avl_node_t *node) {
    if (!root) return node;
    int cmp = key_cmp(node->size, root->size, (uintptr_t)node, (uintptr_t)root);
    if (cmp < 0) {
        root->left = avl_insert(root->left, node);
    } else {
        root->right = avl_insert(root->right, node);
    }
    update_height(root);
    int balance = get_balance(root);

    //ll
    if (balance > 1 && key_cmp(node->size,root->left->size, (uintptr_t)node, (uintptr_t)root->left) < 0)
        return rotate_right(root);
    //rr
    if (balance < -1 && key_cmp(node->size,root->right->size, (uintptr_t)node, (uintptr_t)root->right) > 0)
        return rotate_left(root);
    //lr
    if (balance > 1 && key_cmp(node->size,root->left->size, (uintptr_t)node, (uintptr_t)root->left) > 0) {
        root->left = rotate_left(root->left);
        return rotate_right(root);
    }
    //rl
    if (balance < -1 && key_cmp(node->size,root->right->size, (uintptr_t)node, (uintptr_t)root->right) < 0) {
        root->right = rotate_right(root->right);
        return rotate_left(root);
    }
    return root;
}

static avl_node_t* avl_min_node(avl_node_t *n) {
    avl_node_t *cur = n;
    while (cur && cur->left) cur = cur->left;
    return cur;
}

static avl_node_t* avl_max_node(avl_node_t *n) {
    avl_node_t *cur = n;
    while (cur && cur->right) cur = cur->right;
    return cur;
}

static avl_node_t* avl_remove_key(avl_node_t *root, size_t key_sz, uintptr_t key_addr) {
    if (!root) return NULL;
    int cmp = key_cmp(key_sz,root->size,key_addr,(uintptr_t)root->blk);
    if (cmp < 0){
        root->left = avl_remove_key(root->left,key_sz, key_addr);
    } else if (cmp > 0){
        root->right = avl_remove_key(root->right,key_sz,key_addr);
    } else {
        //found
        if (!root->left || !root->right) { //one child
            avl_node_t* temp = root->left? root->left : root->right;
            if (!temp) {
                munmap((void*)root,sizeof(avl_node_t));
                return NULL;
            } else {
                munmap((void*)root,sizeof(avl_node_t));
                return temp;
            }
        } else {
            //two children.find succesor
            avl_node_t *succ = avl_min_node(root->right);
            root->size = succ->size;
            root->blk = succ->blk;
            root->right = avl_remove_key(root->right,succ->size,(uintptr_t)succ->blk);
        }
    }
    update_height(root);
    int balance = get_balance(root);

    if (balance > 1 && get_balance(root->left) >= 0) return rotate_right(root);
    if (balance > 1 && get_balance(root->left) < 0){
        root->left = rotate_left(root->left);
        return rotate_right(root);
    }
    if (balance < -1 && get_balance(root->right) <= 0) return rotate_left(root);
    if (balance < -1 && get_balance(root->right) > 0){
        root->right = rotate_right(root->right);
        return rotate_left(root);
    }
    return root;
}

static avl_node_t* avl_lower_bound(avl_node_t *root,size_t size) {
    avl_node_t* res = NULL;
    while (root){
        if (root->size >= size){
            res = root;
            root = root->left;
        } 
        else{
            root = root->right;
        }
    }
    return res;
}


// Function to allocate memory using mmap (best-fit strategy)
void* malloc_best_fit(size_t size) {
    // Implementation of best-fit allocation
    if(size==0) return NULL;
    if(head == MAP_FAILED) {
        init_heap();
        mode = BEST_FIT;
        avl_root = avl_node_new(head);
    }
    if (head == NULL) // no meory left
    {
        return NULL;
    }
    size_t size_to_allocate = size + sizeof(header_t);
    avl_node_t* n = avl_lower_bound(avl_root, size_to_allocate-((size_to_allocate > sizeof(node_t))? sizeof(node_t):0));
    if(n){
        node_t* current = n->blk;
        avl_root = avl_remove_key(avl_root, current->size, (uintptr_t)current);
        size_t x = current->size;
        node_t* prev = current->prev;
        node_t* nxt = current->next;
        header_t* header = (header_t*)((char*)current);
        void* ptr = (void*)((char*)header + sizeof(header_t));
        if(x >= size_to_allocate){
            // split
            header->size = size;
            header->magic = MAGIC_NUMBER;
            node_t* new_node = (node_t*)((char*)current+size_to_allocate);
            new_node->next=nxt;
            new_node->prev=prev;
            if(prev) prev->next = new_node;
            else head = new_node;
            if(nxt) nxt->prev = new_node;
            new_node->size=x-size_to_allocate;
            avl_root = avl_insert(avl_root,avl_node_new(new_node));
        }
        else{
            // delete
            header->size = x-sizeof(header_t)+sizeof(node_t);
            header->magic = MAGIC_NUMBER;
            if(prev) prev->next = nxt;
            else head = nxt;
            if(nxt) nxt->prev = prev;
        }
        return ptr;
    }
    return NULL;
}

// Function to allocate memory using mmap (first-fit strategy)
void* malloc_first_fit(size_t size) {
    // Implementation of first-fit allocation
    if(size==0) return NULL;
    if(head == MAP_FAILED) {
        init_heap();
        mode = FIRST_FIT;
    }
    if (head == NULL)
    {
        return NULL;
    }
    
    size_t size_to_allocate = size + sizeof(header_t);
    node_t* current = head;
    node_t* prev = NULL;
    bool del_curr = false;
    while(current != NULL) {
        if(current->size >= size_to_allocate) {
            break;
        }
        // if removing this node will do it. check that node is not head or if it is head then it's next ele is not null
        else if((current->size+sizeof(node_t) >= size_to_allocate)){
            del_curr=true;
            break;
        }
        prev = current;
        current = current->next;
    }
    if(current == NULL) {
        return NULL;
    }
    node_t* nxt = current->next;
    size_t x = current->size;
    // allocatate at block pointed by current
    header_t* header = (header_t*)((char*)current);
    header->size = x+sizeof(node_t)-sizeof(header_t);
    header->magic = MAGIC_NUMBER;
    void* ptr = (void*)((char*)current + sizeof(header_t));
    if(!del_curr){
        header->size = size;
        node_t* new_node = (node_t*)((char*)current+size_to_allocate);
        new_node->prev = prev;
        new_node->next=nxt;
        if(nxt) nxt->prev = new_node;
        new_node->size = x - size_to_allocate;
        if(prev!=NULL){
            prev->next=new_node;
        }
        else{
            head = new_node;
        }
    }
    else{
        if(prev == NULL){
            head = nxt;
            if(nxt) nxt->prev = NULL;
        }
        else{
            prev->next = nxt;
            if(nxt) nxt->prev = prev;
        }

    }
    return ptr;
}

// Function to allocate memory using mmap (next-fit strategy)
uintptr_t next_loc = 0;
void* malloc_next_fit(size_t size) {
    // Implementation of next-fit allocation
    if(size==0) return NULL;
    if(head == MAP_FAILED) {
        init_heap();
        mode = NEXT_FIT;
        next_loc = (uintptr_t)head;
    }
    if (head == NULL) // no meory left
    {
        return NULL;
    }
    
    size_t size_to_allocate = size + sizeof(header_t);
    node_t* current = head;
    while (current && (uintptr_t)current < next_loc)
    {
        current = current->next;
    }
    if(!current) current = head;
    node_t* prev = NULL;
    bool del_curr = false;
    while(current != NULL) {
        if(current->size >= size_to_allocate) {
            break;
        }
        // if removing this node will do it. check that node is not head or if it is head then it's next ele is not null
        else if((current->size+sizeof(node_t) >= size_to_allocate)){
            del_curr=true;
            break;
        }
        prev = current;
        current = current->next;
    }
    if(current == NULL) {
        current = head;
        while(current != NULL) {
            if(current->size >= size_to_allocate) {
                break;
            }
            // if removing this node will do it. check that node is not head or if it is head then it's next ele is not null
            else if((current->size+sizeof(node_t) >= size_to_allocate)){
                del_curr=true;
                break;
            }
            prev = current;
            current = current->next;
        }
        if(!current) return NULL;
    }
    node_t* nxt = current->next;
    size_t x = current->size;
    // allocatate at block pointed by current
    header_t* header = (header_t*)((char*)current);
    header->size = x+sizeof(node_t)-sizeof(header_t);
    header->magic = MAGIC_NUMBER;
    void* ptr = (void*)((char*)current + sizeof(header_t));
    next_loc = (uintptr_t)header + x + sizeof(node_t);
    if(!del_curr){
        header->size = size;
        node_t* new_node = (node_t*)((char*)current+size_to_allocate);
        new_node->prev = prev;
        new_node->next=nxt;
        if(nxt) nxt->prev = new_node;
        new_node->size = x - size_to_allocate;
        if(prev!=NULL){
            prev->next=new_node;
        }
        else{
            head = new_node;
        }
        next_loc = (uintptr_t)(new_node);
    }
    else{
        if(prev == NULL){
            head = nxt;
            if(nxt) nxt->prev = NULL;
        }
        else{
            prev->next = nxt;
            if(nxt) nxt->prev = prev;
        }
    }
    return ptr;
}



// Function to allocate memory using mmap (worst-fit strategy)
void* malloc_worst_fit(size_t size) {
    // Implementation of worst-fit allocation
    if(size==0) return NULL;
    if(head == MAP_FAILED) {
        init_heap();
        mode = WORST_FIT;
        avl_root = avl_node_new(head);
    }
    if (head == NULL) // no meory left
    {
        return NULL;
    }
    size_t size_to_allocate = size + sizeof(header_t);
    avl_node_t* n = avl_max_node(avl_root);
    if(!n) return NULL;
    if(n->size+sizeof(node_t) < size_to_allocate) return NULL;
    if(n){
        node_t* current = n->blk;
        avl_root = avl_remove_key(avl_root, current->size, (uintptr_t)current);
        size_t x = current->size;
        node_t* prev = current->prev;
        node_t* nxt = current->next;
        header_t* header = (header_t*)((char*)current);
        void* ptr = (void*)((char*)header + sizeof(header_t));
        if(x >= size_to_allocate){
            // split
            header->size = size;
            header->magic = MAGIC_NUMBER;
            node_t* new_node = (node_t*)((char*)current+size_to_allocate);
            new_node->next=nxt;
            new_node->prev=prev;
            if(prev) prev->next = new_node;
            else head = new_node;
            if(nxt) nxt->prev = new_node;
            new_node->size=x-size_to_allocate;
            avl_root = avl_insert(avl_root,avl_node_new(new_node));
        }
        else{
            // delete
            header->size = x-sizeof(header_t)+sizeof(node_t);
            header->magic = MAGIC_NUMBER;
            if(prev) prev->next = nxt;
            else head = nxt;
            if(nxt) nxt->prev = prev;
        }
        return ptr;
    }
    return NULL;
}
uintptr_t base =0;
// Function to allocate memory using mmap (buddy allocation strategy)
void* malloc_buddy_alloc(size_t size) {
    if (size == 0) return NULL;
    if (head == MAP_FAILED) {
        init_heap();
        head->size = BLOCK_SIZE;
        base = (uintptr_t)head;
        mode = BUDDY;
    }
    if (head == NULL) return NULL;

    size_t size_to_allocate = size + sizeof(header_t);

    size_t target = 1;// target size of block. to find
    while (target < size_to_allocate) target <<= 1;
    if (target < sizeof(node_t)) target = sizeof(node_t);

    // find smallest free block >= target (best-fit among powers of two)
    node_t *cur = head;
    node_t *best = NULL;
    while (cur) {
        if (cur->size >= target) {
            if (best == NULL || (cur->size < best->size)) best = cur;
        }
        cur = cur->next;
    }
    if (best == NULL){
        return NULL;
    } 
        

    //split until block size is target
    while (best->size > target) {
        size_t half = (best->size) / 2;
        node_t *buddy = (node_t*)((char*)best + half);
        buddy->size = half;
        buddy->next = best->next;
        buddy->prev = best;
        if (best->next) best->next->prev = buddy;
        best->next = buddy;
        best->size = half;
    }
    size_t allocated = best->size;
    if (best->prev) best->prev->next = best->next;
    else head = best->next;
    if (best->next) best->next->prev = best->prev;

    header_t *header = (header_t*)((char*)best);
    header->size = allocated-sizeof(header_t);
    header->magic = MAGIC_NUMBER;
    void *ptr = (void*)((char*)header + sizeof(header_t));
    return ptr;
}

// Function to release memory allocated using your malloc functions
void my_free(void* ptr) {
    // Implementation of memory freeing
    if(ptr == NULL) return;
    header_t* header = (header_t*)((char*)ptr-sizeof(header_t));
    if(header->magic != MAGIC_NUMBER){
        fprintf(stderr, "Double free");
        abort();
    } 
    header->magic = 0;
    size_t size_allocated = header->size+sizeof(header_t);
    if(mode == FIRST_FIT || mode == NEXT_FIT){
        if(head == NULL){
            head = (node_t*)(header);
            head->prev = NULL;
            head->next = NULL;
            head->size = size_allocated-sizeof(node_t);
            return;
        }
        uintptr_t curr_addr = (uintptr_t)head;
        node_t* current = head;
        uintptr_t header_addr = (uintptr_t)header;
        if(header_addr < curr_addr){
            node_t* new_node = (node_t*)header;
            new_node->prev = NULL;
            if(header_addr+size_allocated == curr_addr){
                new_node->next = head->next;
                new_node->size = head->size + size_allocated;
                if(head->next) head->next->prev = new_node;
            }
            else {
                new_node->next = head;
                new_node->size = size_allocated - sizeof(node_t);
                head->prev = new_node;
            }
            head = new_node;
            return;
        }
        node_t* prev = NULL;
        while (current && curr_addr < header_addr)
        {
            prev = current;
            current = current->next;
            curr_addr = (uintptr_t)current;
        }
        if(curr_addr == 0){ // place at last position
            current = prev;
            curr_addr = (uintptr_t)current;
            if((curr_addr+current->size+sizeof(node_t)) == header_addr){
                current->size += size_allocated;
                return;
            }
            node_t* new_node = (node_t*)header;
            new_node->prev = current;
            new_node->next = NULL;
            new_node->size = size_allocated - sizeof(node_t);
            current->next = new_node;
            return;
        }
        else{
            if((prev->size+sizeof(node_t)+(uintptr_t)prev == header_addr) && (curr_addr-size_allocated == header_addr)){
                // prev.size, prev.next, prev.next.prev
                prev->size += size_allocated+sizeof(node_t)+current->size;
                prev->next = current->next;
                if(prev->next)prev->next->prev = prev;
            }
            else if(prev->size+sizeof(node_t)+(uintptr_t)prev == header_addr){
                prev->size += size_allocated;
            }
            else if(curr_addr-size_allocated == header_addr){
                node_t* new_node = (node_t*)header;
                new_node->size = current->size + size_allocated; 
                new_node->prev = prev;
                prev->next = new_node;
                new_node->next = current->next;
                if(current->next) current->next->prev = new_node;
            }
            else{
                node_t* new_node = (node_t*)header;
                new_node->next = current;
                current->prev = new_node;
                new_node->prev = prev;
                prev->next = new_node;
                new_node->size = size_allocated-sizeof(node_t);
            }
        }
        return;
    }
    if(mode == WORST_FIT || mode == BEST_FIT){

        if(head == NULL){
            head = (node_t*)(header);
            head->prev = NULL;
            head->next = NULL;
            head->size = size_allocated-sizeof(node_t);
            avl_root = avl_node_new(head);
            return;
        }
        uintptr_t curr_addr = (uintptr_t)head;
        node_t* current = head;
        uintptr_t header_addr = (uintptr_t)header;
        if(header_addr < curr_addr){
            node_t* new_node = (node_t*)header;
            new_node->prev = NULL;
            avl_root = avl_remove_key(avl_root, head->size, (uintptr_t)head);
            if(header_addr+size_allocated == curr_addr){// coalese new_node+head –
                new_node->next = head->next;
                new_node->size = head->size + size_allocated;
                if(head->next) head->next->prev = new_node;
                head = new_node;
                avl_root = avl_insert(avl_root, avl_node_new(head));
            }
            else {//new_node – head –
                new_node->next = head;
                new_node->size = size_allocated - sizeof(node_t);
                head->prev = new_node;
                avl_root = avl_insert(avl_root, avl_node_new(head));
                // head – old_head – to maintain order
                head = new_node;
                avl_root = avl_insert(avl_root, avl_node_new(head));
            }
            return;
        }
        node_t* prev = NULL;
        while (current && curr_addr < header_addr)
        {
            prev = current;
            current = current->next;
            curr_addr = (uintptr_t)current;
        }
        if(curr_addr == 0){ // last
            current = prev;
            curr_addr = (uintptr_t)current;
            if(curr_addr+current->size+sizeof(node_t) == header_addr){// coalese current – null
                avl_root = avl_remove_key(avl_root, current->size, curr_addr);
                current->size += size_allocated;
                avl_root = avl_insert(avl_root, avl_node_new(current));
            }
            else{
                node_t* new_node = (node_t*)header;
                new_node->prev = current;
                new_node->next = NULL;
                new_node->size = size_allocated - sizeof(node_t);
                current->next = new_node;
                avl_root = avl_insert(avl_root, avl_node_new(new_node));
            }

        }
        else{
            if((prev->size+sizeof(node_t)+(uintptr_t)prev == header_addr) && (curr_addr-size_allocated == header_addr)){
                // prev.size, prev.next, prev.next.prev
                avl_root = avl_remove_key(avl_root, prev->size, (uintptr_t)prev);
                avl_root = avl_remove_key(avl_root, current->size, curr_addr);
                prev->size += size_allocated+sizeof(node_t)+current->size;
                prev->next = current->next;
                if(prev->next) prev->next->prev = prev;
                avl_root = avl_insert(avl_root, avl_node_new(prev));
            }
            else if(prev->size+sizeof(node_t)+(uintptr_t)prev == header_addr){// pre coallese prev_new – current
                avl_root = avl_remove_key(avl_root, prev->size, (uintptr_t)prev);
                prev->size += size_allocated;
                avl_root = avl_insert(avl_root, avl_node_new(prev));
            }
            else if(curr_addr-size_allocated == header_addr){// prev – new_node – current.next
                avl_root = avl_remove_key(avl_root, current->size, curr_addr);
                node_t* new_node = (node_t*)header;
                new_node->size = current->size + size_allocated; 
                new_node->prev = prev;
                prev->next = new_node;
                new_node->next = current->next;
                if(new_node->next) new_node->next->prev = new_node;
                avl_root = avl_insert(avl_root, avl_node_new(new_node));
            }
            else{// prev – new_node – current
                node_t* new_node = (node_t*)header;
                new_node->next = current;
                current->prev = new_node;
                new_node->prev = prev;
                prev->next = new_node;
                new_node->size = size_allocated-sizeof(node_t);
                avl_root = avl_insert(avl_root, avl_node_new(new_node));
            }
        }
    }
    if(mode == BUDDY){

        if(head == NULL){
            head = (node_t*)header;
            head->size = size_allocated;
            head->next = head->prev = NULL;
            return;
        }

        size_t offset = (uintptr_t)header-base;
        node_t* bud = (node_t*)(base+offset^size_allocated);
        node_t* current = NULL;
        if (((header_t*)bud)->magic != MAGIC_NUMBER){// first coalese
            if(offset & size_allocated){ // hdr is right of buddy
                bud->size <<= 1;
                current = bud;
            }
            else{ // hdr is left of buddy
                node_t* new_node = (node_t*)header;
                new_node->size = (size_allocated << 1);
                new_node->next = bud->next;
                if(new_node->next) new_node->next->prev = new_node;
                new_node->prev = bud->prev;
                if(new_node->prev) new_node->prev->next = new_node;
                current = new_node;
                if((uintptr_t)head > (uintptr_t)current){
                    head=current;
                }
            }
        }
        else{ // no coaleseing just place in sorted order
            node_t* new_node = (node_t*)header;
            new_node->size = size_allocated;
            current = head;
            node_t* prev = NULL;
            while (current && (uintptr_t)current < (uintptr_t)new_node)
            {
                prev = current;
                current = current->next;
            }
            new_node->prev = prev;
            if(prev) prev->next = new_node;
            else head = new_node;
            new_node->next = current;
            if(current) current->prev = new_node;
            return;
        }
        if(current->size == BLOCK_SIZE){
            head = current;
            return;
        }
        // further coalesing
        // grow from current
        offset = (uintptr_t)current - base;
        bud = (node_t*)(base + offset^(current->size));
        while (((header_t*)bud)->magic != MAGIC_NUMBER){
            if(offset & current->size){ // current is right of buddy. keep buddy, remove current
                bud->size <<= 1;
                bud->next = current->next;
                if(bud->next) bud->next->prev = bud;
                current = bud;
                if(current->size == BLOCK_SIZE){
                    head = bud;
                    return;
                }
                offset = (uintptr_t)current - base;
                bud = (node_t*)(base + offset^(current->size));
            }
            else{ // current is left of buddy. keep current, remove bud. 
                current->size <<= 1;
                current->next = bud->next;
                if(current->next) current->next->prev = current;
                current->prev = bud->prev;
                if(current->prev) current->prev->next = current;
                if(current->size == BLOCK_SIZE){
                    head = current;
                    return;
                }
                offset = (uintptr_t)current - base;
                bud = (node_t*)(base + offset^(current->size));
            }
        }
    }
    if(mode == 0){
        fprintf(stderr,"Pointer not on heap.");
        abort();
    }
}
