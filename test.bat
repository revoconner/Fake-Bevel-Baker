@REM  --seed · --format png|exr · --out-world
 
 bebvenv/Scripts/python.exe -m fake_bevel_baker.main --mesh tests/fixtures/sphere.obj --out tests/fixtures/sphere.png --resolution 4096 --samples 32 --angle 45 --radius 3.0 