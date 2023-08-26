# badgrad

Like [**micrograd**](https://github.com/karpathy/micrograd) or [**tinygrad**](https://github.com/geohot/tinygrad), but trash.

This is the most basic possible collection of code that can somehow qualify as a NN framework.

### Goal

make mnist classifier using no libraries 

![](https://raw.githubusercontent.com/An-GG/badgrad/master/visual.gif)

### Stretch Goal

do it in vanilla
- ts
- python
- c
- swift


make it multithreaded


### Major Stretch Goal

make it not trash




## ok


nodes are values
lines are mutltipliers

nodes sum thier inputs, then sigmoid 
lines multiply by bias 


### how to train

- set input to test labeled data, calculate all
- compare to ideal output. Desired - Actual = Diff
- decide Stablitiy (0,1), how much biases change
- Each node has a Diff. ScaleFactor = 1 / (1 - Stability * Diff)
- node=0.8, should be 0.0, Diff=-0.8 SF = 1 / (1 + 0.5 * 0.8) = 0.714
- For each node:
        - For each input line 
            - multiply bias by Diff*Stability
            - 1 / (1 - that) = ScaleFactor
            - multiply each input bias by that

This way, if one node has a diff of +0.3 but another -0.3, 

cool to see it still works

```
# git foolproof thing to clone all submodules recursively
git submodule update --init --recursive

npm install
```

then in one terminal window 1
```
npx ts-node grad.ts
```

then to monitor progress,


other terminal window (2):
```
cd viewer
npx tsc -w
```

window (3)
```
cd viewer
npx http-server
```


---

to get a progress update:


window (4)
```
npx ts-node compositifiy_netfiles.ts && open "http://localhost:8080/"
```
& hit the play button <br><br>

i ran it for a bit on a m2 macbook air and saved the progress report from temp_netfiles in the [LFB](https://github.com/An-GG/LFB) folder on the 'm2-air-progress-report' branch as temp_netfiles.zip.d39eddebecd134708e9efcde7d10ce13652e0c63767a16753cc202b76585441e.parts 

