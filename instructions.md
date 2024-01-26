Steps to be done in order:
1. install TrackMania 2020
   - C:\Users\username\Documents\Trackmania appeared
2. install Openplanet 1.26.9
   - C:\Users\username\OpenplanetNext appeared
   - in-game layout appeared
   - TMRL_GrabData.op plugin appeared and connects
3. create conda env with install_reqs from tmrl/setup.py
    - pytorch from console with args as on their website
4. conda install pywin32
5. pip install tmrl (make the controller and configs work)
   - C:\Users\username\TmrlData appeared (params config for the package)
   - only after this step a copy of the tmrl project compiles (hard paths, 
     still needs a correct call to set off)
6. map tmrl-test.Map.Gbx copied to C:\Users\username\Documents\Trackmania\Maps\My Maps 
   (appeared in-game)
7. update C:\Users\username\TmrlData\config\config.json to match a desired model 
   (check tmrl\readme\get_started.md)
8. turn on the game, adjust it to the left-top corner, turn on the map and set the racing mode
9. open the map, set up the camera and turn off the ghost (NUM_3 twice for LIDAR)
10. run 'python -m tmrl --test' in anaconda prompt in your env, focus the window on the game

Training own reward fn:
1. make sure a model has a unique name! + everything above checks
2. run 'python -m tmrl --server', 'python -m tmrl --trainer' and 'python -m tmrl --worker'
   from anaconda prompts, focus the window on the game