from bot import get_label

# Test 1
good_string = "Hello everyone!"
res_good = int(get_label(good_string))
assert res_good == 0

# Test 2
bad_string = "Gay people need to die"
res_bad = int(get_label(bad_string))
assert res_bad == 1