import xml.etree.ElementTree as ET
import os

def create_nlos_only_version(input_file, output_file):
    # 1. Parse the input XML
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    tree = ET.parse(input_file)
    root = tree.getroot()

    # 2. Find the integrator
    integrator = root.find("integrator")
    if integrator is None:
        print("Error: No <integrator> tag found in the scene.")
        return

    # 3. Locate the 'use_nlos_only' boolean and change it to true
    # We iterate through children to ensure we find the specific named boolean
    param_found = False
    for child in integrator.findall("boolean"):
        if child.get("name") == "use_nlos_only":
            print(f"Found parameter 'use_nlos_only'. Changing value from '{child.get('value')}' to 'true'.")
            child.set("value", "true")
            param_found = True
            break
    
    if not param_found:
        print("Warning: Could not find <boolean name='use_nlos_only'> inside the integrator.")
        # Optional: You could create it if it's missing
        # ET.SubElement(integrator, "boolean", name="use_nlos_only", value="true")

    # 4. Save the new file
    # Adding indentation for readability (Python 3.9+)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="    ", level=0)
        
    tree.write(output_file, encoding="utf-8", xml_declaration=False)
    print(f"Successfully created '{output_file}'")

if __name__ == "__main__":
    # Define your input and desired output filenames
    import argparse

    parser = argparse.ArgumentParser(description="Convert a Mitsuba NLOS transient scene to a NLOS-only version (sets use_nlos_only=True).")
    parser.add_argument("input_file", type=str, help="Input Mitsuba XML scene file.")
    parser.add_argument("output_file", type=str, help="Output XML file with use_nlos_only=true.")

    args = parser.parse_args()

    create_nlos_only_version(args.input_file, args.output_file)