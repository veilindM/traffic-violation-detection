# firebase_utils.py - Final Version with Vehicle Type & Color Support
import firebase_admin
from firebase_admin import credentials, storage, firestore
import os
from datetime import datetime
from config import FIREBASE_KEY_PATH, FIREBASE_BUCKET

# Initialize Firebase app once
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        "storageBucket": FIREBASE_BUCKET
    })

db = firestore.client()
bucket = storage.bucket()

def upload_violation_image(local_path, plate_number, frame_no, vehicle_color="UNKNOWN", vehicle_type="UNKNOWN"):
    """
    Upload the image at local_path to Firebase Storage and
    create a Firestore document in collection 'violations'.
    
    Args:
        local_path: Path to the violation image
        plate_number: Detected license plate number
        frame_no: Frame number where violation occurred
        vehicle_color: Detected vehicle color
        vehicle_type: Detected vehicle type (CAR, TRUCK, BUS, MOTORCYCLE)
    
    Returns:
        Public URL (or storage path)
    """
    blob_name = f"violations/{os.path.basename(local_path)}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    
    # Optionally make public (easy for testing). Production: use signed URLs or secure rules.
    try:
        blob.make_public()
        url = blob.public_url
    except Exception:
        url = f"gs://{FIREBASE_BUCKET}/{blob_name}"
    
    # Store metadata in Firestore (with color and type)
    doc = {
        "frame": frame_no,
        "plate_number": plate_number,
        "vehicle_color": vehicle_color,
        "vehicle_type": vehicle_type,
        "image_url": url,
        "blob_path": blob_name,
        "timestamp": datetime.utcnow()
    }
    
    db.collection("violations").add(doc)
    print(f"[firebase_utils] uploaded {local_path} -> {url} (Type: {vehicle_type}, Color: {vehicle_color})")
    
    return url

def get_all_violations():
    """
    Retrieve all violations from Firestore.
    Returns list of violation dictionaries.
    """
    violations = []
    docs = db.collection("violations").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    
    for doc in docs:
        violation_data = doc.to_dict()
        violation_data['id'] = doc.id
        violations.append(violation_data)
    
    return violations

def get_violations_by_color(color):
    """
    Get violations filtered by vehicle color.
    Useful for searching specific colored vehicles.
    
    Args:
        color: Vehicle color (e.g., 'RED', 'BLUE', 'WHITE')
    
    Returns:
        List of violations with that color
    """
    violations = []
    docs = db.collection("violations").where("vehicle_color", "==", color).stream()
    
    for doc in docs:
        violation_data = doc.to_dict()
        violation_data['id'] = doc.id
        violations.append(violation_data)
    
    return violations

def get_violations_by_type(vehicle_type):
    """
    Get violations filtered by vehicle type.
    
    Args:
        vehicle_type: Vehicle type (CAR, TRUCK, BUS, MOTORCYCLE)
    
    Returns:
        List of violations of that type
    """
    violations = []
    docs = db.collection("violations").where("vehicle_type", "==", vehicle_type).stream()
    
    for doc in docs:
        violation_data = doc.to_dict()
        violation_data['id'] = doc.id
        violations.append(violation_data)
    
    return violations

def get_violation_statistics():
    """
    Get statistical summary of all violations.
    Returns dictionary with color distribution, type stats, etc.
    """
    violations = get_all_violations()
    
    if not violations:
        return {
            "total_violations": 0,
            "color_distribution": {},
            "type_distribution": {},
            "type_color_combinations": {}
        }
    
    # Color and type distribution
    color_counts = {}
    type_counts = {}
    type_color_combos = {}
    
    for v in violations:
        color = v.get('vehicle_color', 'UNKNOWN')
        vtype = v.get('vehicle_type', 'UNKNOWN')
        
        color_counts[color] = color_counts.get(color, 0) + 1
        type_counts[vtype] = type_counts.get(vtype, 0) + 1
        
        combo = f"{color} {vtype}"
        type_color_combos[combo] = type_color_combos.get(combo, 0) + 1
    
    return {
        "total_violations": len(violations),
        "color_distribution": color_counts,
        "type_distribution": type_counts,
        "type_color_combinations": type_color_combos
    }

def search_violation_by_plate(plate_number):
    """
    Search for a specific violation by plate number.
    Returns list of matching violations.
    """
    violations = []
    docs = db.collection("violations").where("plate_number", "==", plate_number).stream()
    
    for doc in docs:
        violation_data = doc.to_dict()
        violation_data['id'] = doc.id
        violations.append(violation_data)
    
    return violations

def search_violations_by_type_and_color(vehicle_type, vehicle_color):
    """
    Search for violations by both type and color.
    Example: Find all RED CARs
    
    Args:
        vehicle_type: Vehicle type (CAR, TRUCK, BUS, MOTORCYCLE)
        vehicle_color: Vehicle color (RED, BLUE, WHITE, etc.)
    
    Returns:
        List of matching violations
    """
    violations = []
    docs = db.collection("violations") \
        .where("vehicle_type", "==", vehicle_type) \
        .where("vehicle_color", "==", vehicle_color) \
        .stream()
    
    for doc in docs:
        violation_data = doc.to_dict()
        violation_data['id'] = doc.id
        violations.append(violation_data)
    
    return violations

def delete_violation(violation_id):
    """
    Delete a violation document and its associated image.
    
    Args:
        violation_id: Firestore document ID
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get violation document
        doc_ref = db.collection("violations").document(violation_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            print(f"[firebase_utils] Violation {violation_id} not found")
            return False
        
        violation_data = doc.to_dict()
        
        # Delete image from Storage
        blob_path = violation_data.get('blob_path')
        if blob_path:
            try:
                blob = bucket.blob(blob_path)
                blob.delete()
                print(f"[firebase_utils] Deleted image: {blob_path}")
            except Exception as e:
                print(f"[firebase_utils] Error deleting image: {e}")
        
        # Delete Firestore document
        doc_ref.delete()
        print(f"[firebase_utils] Deleted violation: {violation_id}")
        
        return True
        
    except Exception as e:
        print(f"[firebase_utils] Error deleting violation: {e}")
        return False

def upload_multiple_violations(violations_list):
    """
    Upload multiple violations at once (batch operation).
    
    Args:
        violations_list: List of dicts with keys: local_path, plate_number, frame_no, vehicle_color, vehicle_type
    
    Returns:
        List of uploaded URLs
    """
    urls = []
    for violation in violations_list:
        try:
            url = upload_violation_image(
                local_path=violation['local_path'],
                plate_number=violation['plate_number'],
                frame_no=violation['frame_no'],
                vehicle_color=violation.get('vehicle_color', 'UNKNOWN'),
                vehicle_type=violation.get('vehicle_type', 'UNKNOWN')
            )
            urls.append(url)
        except Exception as e:
            print(f"[firebase_utils] Error uploading {violation.get('local_path')}: {e}")
            urls.append(None)
    
    return urls

def print_violation_stats():
    """
    Print a summary of all violations (useful for debugging).
    """
    stats = get_violation_statistics()
    
    print("\n" + "="*60)
    print("📊 FIREBASE VIOLATION STATISTICS")
    print("="*60)
    print(f"Total Violations: {stats['total_violations']}")
    
    if stats['total_violations'] == 0:
        print("No violations found in Firebase.")
        print("="*60 + "\n")
        return
    
    print(f"\n🚗 Vehicle Type Distribution:")
    for vtype, count in sorted(stats['type_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_violations']) * 100
        print(f"   {vtype:12s}: {count:3d} ({percentage:5.1f}%)")
    
    print(f"\n🎨 Color Distribution:")
    for color, count in sorted(stats['color_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_violations']) * 100
        print(f"   {color:12s}: {count:3d} ({percentage:5.1f}%)")
    
    print(f"\n🔍 Top Vehicle Type + Color Combinations:")
    sorted_combos = sorted(stats['type_color_combinations'].items(), key=lambda x: x[1], reverse=True)[:5]
    for combo, count in sorted_combos:
        print(f"   {combo:20s}: {count:3d}")
    
    print("="*60 + "\n")

def get_most_common_violator():
    """
    Find the most common vehicle type and color that violates.
    Returns dict with most common type and color.
    """
    stats = get_violation_statistics()
    
    if stats['total_violations'] == 0:
        return None
    
    # Most common type
    most_common_type = max(stats['type_distribution'].items(), key=lambda x: x[1]) if stats['type_distribution'] else ('UNKNOWN', 0)
    
    # Most common color
    most_common_color = max(stats['color_distribution'].items(), key=lambda x: x[1]) if stats['color_distribution'] else ('UNKNOWN', 0)
    
    # Most common combination
    most_common_combo = max(stats['type_color_combinations'].items(), key=lambda x: x[1]) if stats['type_color_combinations'] else ('UNKNOWN', 0)
    
    return {
        'most_common_type': most_common_type[0],
        'type_count': most_common_type[1],
        'most_common_color': most_common_color[0],
        'color_count': most_common_color[1],
        'most_common_combination': most_common_combo[0],
        'combination_count': most_common_combo[1]
    }

# Testing/Usage Examples
if __name__ == "__main__":
    print("Firebase Utils - Testing Functions\n")
    
    # Example 1: Get all violations
    print("1. Getting all violations...")
    all_violations = get_all_violations()
    print(f"   Found {len(all_violations)} violations\n")
    
    # Example 2: Get statistics
    print("2. Getting statistics...")
    print_violation_stats()
    
    # Example 3: Search by type
    print("3. Searching for CAR violations...")
    car_violations = get_violations_by_type('CAR')
    print(f"   Found {len(car_violations)} car violations\n")
    
    # Example 4: Search by color
    print("4. Searching for RED vehicle violations...")
    red_violations = get_violations_by_color('RED')
    print(f"   Found {len(red_violations)} red vehicle violations\n")
    
    # Example 5: Most common violator
    print("5. Finding most common violator...")
    common = get_most_common_violator()
    if common:
        print(f"   Most common type: {common['most_common_type']} ({common['type_count']} violations)")
        print(f"   Most common color: {common['most_common_color']} ({common['color_count']} violations)")
        print(f"   Most common combo: {common['most_common_combination']} ({common['combination_count']} violations)")
    else:
        print("   No violations found")
    print()