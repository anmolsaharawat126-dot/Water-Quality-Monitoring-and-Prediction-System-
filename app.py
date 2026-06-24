print("WATER QUALITY CHECK SYSTEM")
print("--------------------------")

# taking input from user
name = input("Enter sample location name: ")

ph = float(input("Enter pH value: "))
turbidity = float(input("Enter turbidity value: "))
do = float(input("Enter dissolved oxygen: "))
heavy = float(input("Enter heavy metal value: "))

print("\nChecking water quality...\n")

# simple logic (no AI, just if-else)

if ph >= 6.5 and ph <= 8.5:
    ph_status = "OK"
else:
    ph_status = "NOT OK"

if turbidity < 5:
    turb_status = "OK"
else:
    turb_status = "NOT OK"

if do >= 5:
    do_status = "OK"
else:
    do_status = "NOT OK"

if heavy < 0.1:
    heavy_status = "OK"
else:
    heavy_status = "NOT OK"

print("RESULT FOR:", name)
print("--------------------------")
print("pH Status:", ph_status)
print("Turbidity Status:", turb_status)
print("Dissolved Oxygen Status:", do_status)
print("Heavy Metals Status:", heavy_status)

# final decision
if ph_status == "OK" and turb_status == "OK" and do_status == "OK" and heavy_status == "OK":
    print("\nFINAL RESULT: WATER IS SAFE ✅")
else:
    print("\nFINAL RESULT: WATER IS NOT SAFE ❌")
