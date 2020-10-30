package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
    "os"
    "strconv"
    "strings"
    "sync"
    "time"
)

var bias int;
var ppath string;
func Abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
func check(e error) {
    if e != nil {
        panic(e)
    }
}

func find_path(ID int,start string, end string,g *graph,wg *sync.WaitGroup) []string {
    defer wg.Done()
    var bias_local int = bias;
    var bias_update int = 0;
    var cost int = 0;
    var path []string;
    var length = 0 ;
    for {
        cost,path,length = g.getPath(start, end, bias)
        bias_update = cost/length
        if Abs(bias_update-bias_local) > 10 {
            bias_local = bias_update
        } else {
            break;
        }
    }
    s := strconv.Itoa(ID)
    ft, _ := os.Create(ppath+"/path"+s)
    defer ft.Close()
    _, err := ft.WriteString(strings.Join(path," "))
    check(err)
    return path
}

func main() {
    bias, _ = strconv.Atoi(os.Args[1])
    ppath = os.Args[2]
    graph := newGraph()
    file, err := os.Open("Edge")
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()
    rd := bufio.NewReader(file)
    for {
        var ID_1,ID_2 string;
        var weight float32;
		line, err := rd.ReadString('\n')
		if err == io.EOF {
			fmt.Print(line)
			break
		}
        fmt.Sscanf(line,"%s %s %4f\n",&ID_1,&ID_2,&weight);
        var w = int(weight*100);
        graph.addEdge(ID_1,ID_2,w);
    }
    t1 := time.Now()
    task, _ := os.Open("task")
    defer task.Close()
    task_rd := bufio.NewReader(task)
    var count int = 0;
    var wg sync.WaitGroup
    for {
        var start,end string
        line, err := task_rd.ReadString('\n')
        fmt.Sscanf(line,"%s %s \n",&start,&end)
        count = count + 1
        wg.Add(1)
        fmt.Println(count)
        fmt.Println(start,end)
        go find_path(count,start,end,graph,&wg)
        if err == io.EOF {
            fmt.Println(line)
            break
        }
    }
    wg.Wait()
    elapsed := time.Since(t1)
    fmt.Println("Using: ", elapsed)

}
