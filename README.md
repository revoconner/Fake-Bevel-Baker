# Low Poly Fake Bevel Baker

A standalone tool to bake bevels or rounded edges for low poly hard surface models, without needing high poly models. 

<img width="1800" height="auto" alt="2" src="https://github.com/user-attachments/assets/442a6197-fc77-4ad1-991f-71ac627c5532" /> <br>
<img width="1800" height="auto" alt="1" src="https://github.com/user-attachments/assets/32c2b253-1d78-4e27-a55c-569db02eb3af" /> <br>
<img width="1000" height="auto" alt="image" src="https://github.com/user-attachments/assets/40c7203c-0e47-4615-97c1-161083b85236" />

## How to use

#### Windows users can download the packaged tool and run it using the .exe file.

1. Clone the repo
2. Create a virtual environment (recommended) and install these dependencies:
```pip
colorama==0.4.6
embreex==2.17.7.post7
ImageIO==2.37.3
iniconfig==2.3.0
llvmlite==0.47.0
numba==0.65.0
numpy==2.4.4
OpenEXR==3.4.10
packaging==26.1
pillow==12.2.0
pluggy==1.6.0
Pygments==2.20.0
PySide6==6.11.0
pytest==9.0.3
scipy==1.17.1
trimesh==4.11.5
```
3. Activate the environment and type the following to open the GUI: `pythonw.exe -m fake_bevel_baker.ui`

### CLI and flags
```powershell
python.exe -m fake_bevel_baker.main
--mesh "path to mesh file" 
--out "path to baked normal map file"."choose either png or .exr"
--resolution 4096 
--samples "higher is better but takes nore time" 
--angle "keep it exactly as your smoothing group, soften poly edge angle"
--radius "in unit of file"
--seed "is deterministic"
--format ".png|.exr, if defining here keep out to only filename without extension"
--out-world "world space normal map"
```

**EXAMPLE**

`python.exe -m fake_bevel_baker.main --mesh C:/tests/fixtures/sphere.obj --out C:/tests/fixtures/sphere.png --resolution 4096 --samples 32 --angle 45 --radius 3.0 `

## How does it work?

For an indepth explanation of how does it work, read [this.](https://raw.githubusercontent.com/revoconner/fake-bevel-baker/refs/heads/main/explanation.md)
