package main

import hp "container/heap"

type path struct {
    value int
    nodes []string
}


type minPath []path

func (h minPath) Len() int           { return len(h) }
//add bias to get weight
//weight = (norm(color1-color2)-k)^2
func (h minPath) Less(i, j int) bool { 
    return (h[i].value-bias)*(h[i].value-bias)  < (h[j].value-bias)*(h[j].value-bias)
}
func (h minPath) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *minPath) Push(x interface{}) {
    *h = append(*h, x.(path))
}

func (h *minPath) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

type heap struct {
    values *minPath
}

func newHeap() *heap {
    return &heap{values: &minPath{}}
}

func (h *heap) push(p path) {
    hp.Push(h.values, p)
}

func (h *heap) pop() path {
    i := hp.Pop(h.values)
    return i.(path)
}

