from asteroid import ConvTasNet

model_path = 'groadabike/ConvTasNet_DAMP-VSEP_enhboth'

model = ConvTasNet.from_pretrained(model_path)

# get the test file from console
test_path = input("enter the path of the file to test:")

# use the model to separate a file
model.separate(test_path, resample=True, force_overwrite=True)

# display sounds
# display(Audio(test_path))
# display(Audio(test_path.split(".")[0] + '_est1.wav'))
# display(Audio(test_path.split(".")[0] + '_est2.wav'))
print(test_path.split(".")[0] + '_est1.wav')
print(test_path.split(".")[0] + '_est2.wav')
