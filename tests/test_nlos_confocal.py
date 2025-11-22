import mitsuba as mi
import drjit as dr
import pytest
import sys
import os

# Add the parent directory to the path so we can import mitransient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
except Exception:
    pass

def test_nlos_confocal_rendering():
    """
    Test that the transient_nlos_path integrator can be instantiated with
    nlos_confocal=True and use_nlos_only=True, and that it renders without error.
    """
    
    # Define a simple scene
    scene_dict = {
        "type": "scene",
        "integrator": {
            "type": "transient_nlos_path",
            "max_depth": 2,
            "nlos_laser_sampling": True,
            "nlos_confocal": True,
            "use_nlos_only": True,
            "temporal_filter": "box",
        },
        "sensor": {
            "type": "nlos_capture_meter",
            "sensor_origin": [0, 0, 0],
            "laser_origin": [0, 0, 0],
            "laser_lookat_pixel": [16, 16, 0],
            "film": {
                "type": "transient_hdr_film",
                "width": 32,
                "height": 32,
                "temporal_bins": 10,
                "start_opl": 0.0,
                "bin_width_opl": 0.1,
                "rfilter": {
                    "type": "box"
                }
            },
            "sampler": {
                "type": "independent",
                "sample_count": 1
            }
        },
        # Relay wall
        "wall": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate([0, 0, 2]).scale(2),
            "bsdf": {
                "type": "diffuse"
            }
        },
        # Hidden object
        "hidden_object": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate([0, 1, 1]).scale(0.2),
            "bsdf": {
                "type": "diffuse"
            }
        }
    }

    try:
        scene = mi.load_dict(scene_dict)
        sensor = scene.sensors()[0]
        
        # Render the scene
        # We just check if it runs without crashing
        scene.integrator().render(scene, sensor)
        print("Rendering completed successfully with nlos_confocal=True and use_nlos_only=True")
        
    except Exception as e:
        pytest.fail(f"Rendering failed with error: {e}")

if __name__ == "__main__":
    test_nlos_confocal_rendering()
