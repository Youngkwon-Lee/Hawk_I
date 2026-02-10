"""Check Leg Agility MediaPipe detection rate"""

# Count files
train_failed_count = len(open('C:/Users/YK/tulip/Hawkeye/data/leg_agility_train_failed.txt').readlines())
test_failed_count = len(open('C:/Users/YK/tulip/Hawkeye/data/leg_agility_test_failed.txt').readlines())

# Load successful samples
import pickle
with open('C:/Users/YK/tulip/Hawkeye/data/leg_agility_train_v2.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('C:/Users/YK/tulip/Hawkeye/data/leg_agility_test_v2.pkl', 'rb') as f:
    test_data = pickle.load(f)

train_success = len(train_data['X'])
test_success = len(test_data['X'])

print("="*60)
print("Leg Agility MediaPipe Detection Rate")
print("="*60)

print("\nTRAIN SET:")
print(f"  Total videos: {train_success + train_failed_count}")
print(f"  Successful: {train_success}")
print(f"  Failed: {train_failed_count}")
print(f"  Success rate: {train_success / (train_success + train_failed_count) * 100:.1f}%")

print("\nTEST SET:")
print(f"  Total videos: {test_success + test_failed_count}")
print(f"  Successful: {test_success}")
print(f"  Failed: {test_failed_count}")
print(f"  Success rate: {test_success / (test_success + test_failed_count) * 100:.1f}%")

print("\nOVERALL:")
total_videos = train_success + train_failed_count + test_success + test_failed_count
total_success = train_success + test_success
total_failed = train_failed_count + test_failed_count
print(f"  Total videos: {total_videos}")
print(f"  Successful: {total_success}")
print(f"  Failed: {total_failed}")
print(f"  Success rate: {total_success / total_videos * 100:.1f}%")

print("\n" + "="*60)
