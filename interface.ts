import fs from 'fs';
import { promisify } from 'util';


export class MnistReader<T extends "LABELS" | "IMAGES"> {

    public setType: T;
    public metadata: T extends "LABELS" ? 
        {magicN:number, length:number} : 
        {magicN:number, length:number, res: {x:number, y:number, n_pixels:number} } = {} as any;
    private data:Buffer;
    private byteIndex = 0;

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

    private constructor(set: "TRAINING" | "TEST", type: T) {
        this.setType = type;
        this.data = fs.readFileSync(this.fpaths.REL + this.fpaths[set][type]);
    }
    
    private readForward(nBytes: 1 | 4):number {
        this.byteIndex += nBytes;
        if (nBytes == 1) {
            return this.data.readUInt8(this.byteIndex - 1);
        } else {
            return this.data.readInt32BE(this.byteIndex - 4);
        } 
    }

    public static async getReader<SetType extends "LABELS" | "IMAGES">(set: "TRAINING" | "TEST", type: SetType):Promise<MnistReader<SetType>> {
        let r = new MnistReader<SetType>(set, type);

        await r.setup();
        

        
        (r as MnistReader<"LABELS">).metadata = {
            magicN: r.readForward(4),
            length: r.readForward(4)
        }
        if (type == "IMAGES") {
            (r as MnistReader<"IMAGES">).metadata.res = {
                y: r.readForward(4),
                x: r.readForward(4),
                n_pixels:0
            };
            
            (r as MnistReader<"IMAGES">).metadata.res.n_pixels = 
                (r as MnistReader<"IMAGES">).metadata.res.x * (r as MnistReader<"IMAGES">).metadata.res.y;
        }
        console.log(r.metadata);
        return r;
    }

    private async setup() {
    }

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

async function printNth(n:number, to?:number) {
    let img = await MnistReader.getReader("TRAINING", "IMAGES");
    let lbl = await MnistReader.getReader("TRAINING", "LABELS");
    let i = 0;
    while (i < n) {
        img.next();
        lbl.next();
        i++;    
    }
    let end = to ? to : n+1;
    while (i < end) {
        let x = img.vectorToASCII(img.next());
        console.log(x);
        console.log(lbl.next());
        i++;
    }
}


//printNth(160, 175);
