# Path Find
## Prepare
### Build Graph
Need the "Edge" file to build the graph, format as:
> ID_from ID_to Weight

In this task, treat as 
> piexl_ID piexl_ID Color_distance_norm2

### Find task
Need to determine which path need to find,format as:
> ID_from ID_to

as show in "task"
## Run
build as
> go build -o Path_find . 

run as:
> Path_find k

Where k is the start value of the first path.

also can adjust how many cpu you want to use.

## Output
Output files as
> path1,path2......

The path_i means the optimization path of task_i

