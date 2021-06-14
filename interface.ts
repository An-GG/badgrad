import fs from 'fs';
import { promisify } from 'util';

export type MetadataTypes = {
    LABELS: {magicN:number, length:number},
    IMAGES: {magicN:number, length:number, res: {x:number, y:number, n_pixels:number} }
}

export class MnistReader<T extends ("IMAGES" | "LABELS")> {

    public setType: T;
    public metadata: MetadataTypes[T]; 

    private data:Buffer;
    private byteIndex = 0;
    private headerOffset:number;
    private bytesPerObject:number;
    private fpaths = {
        "REL":"mnist/",
        "TRAINING": {
            "IMAGES":"train-images.idx3-ubyte",
            "LABELS":"train-labels-idx1-ubyte"
        },
        "TEST": {
            "IMAGES":"t10k-images.idx3-ubyte",
            "LABELS":"t10k-labels.idx1-ubyte"
        }
    } as const;

    public constructor(set: "TRAINING" | "TEST", type: T) {
        this.setType = type;
        this.data = fs.readFileSync(this.fpaths.REL + this.fpaths[set][type]);
        this.metadata = {} as any;
        if (type == 'IMAGES') {
            let mdref = this.metadata as MetadataTypes['IMAGES'];
            mdref.magicN = this.readForward(4);
            mdref.length = this.readForward(4);
            mdref.res = {
                y: this.readForward(4),
                x: this.readForward(4),
                n_pixels: 0
            }
            mdref.res.n_pixels = mdref.res.y * mdref.res.x;
            this.bytesPerObject = mdref.res.n_pixels;
        } else {
            let mdref = this.metadata as MetadataTypes['LABELS'];
            mdref.magicN = this.readForward(4);
            mdref.length = this.readForward(4);
            this.bytesPerObject = 1;
        }
        this.headerOffset = JSON.parse(JSON.stringify(this.byteIndex));
    }

    /** Read next n bytes and return as BE int **/    
    private readForward(nBytes: 1 | 4):number {
        this.byteIndex += nBytes;
        if (nBytes == 1) {
            return this.data.readUInt8(this.byteIndex - 1);
        } else {
            return this.data.readInt32BE(this.byteIndex - 4);
        } 
    }

    /** Automatically read and return Label or Image data **/
    public next():  T extends "LABELS" ? number : number[] {
        if (this.setType == "LABELS") {
            return (this.readForward(1)) as any;
        } else {
            let v:number[] = [];
            let n_px = (this as MnistReader<"IMAGES">).metadata.res.n_pixels;
            for (let i = 0; i < n_px; i++) {
                v.push(this.readForward(1));
            }
            return v as any;
        }
    }

    /** Move the current index to some position (nth Label/Image) **/
    public setHeadPosition(n:number) {
        this.byteIndex = this.headerOffset + (this.bytesPerObject * n);
    }

    /** Get current head position (nth Label/Image) **/
    public getHeadPosition():number {
        return (this.byteIndex - this.headerOffset) / this.bytesPerObject;
    }

    public vectorToASCII(v: number[]):string {
        const ASCIIScale = " .:-=+*#%@".split('');
        let out = "";
        for (let i=0; i<v.length; i++) { 
            // map 10 to 8bit 256
            let p = v[i];
            out+=(ASCIIScale[Math.floor(p * (10/256))]);
            if ((i+1)%(this as MnistReader<"IMAGES">).metadata.res.x == 0) {
                out+="\n";
            }
        }
        return out;
    } 
}

function printNth(n:number, to?:number) {
    let img = new MnistReader("TRAINING", "IMAGES");
    let lbl = new MnistReader("TRAINING", "LABELS");
    
    img.setHeadPosition(n);
    lbl.setHeadPosition(n);
    
    let i = n;
    let end = to ? to : n+1;
    while (i < end) {
        let x = img.vectorToASCII(img.next());
        console.log(x);
        console.log(lbl.next());
        i++;
    }
}

printNth(0);
