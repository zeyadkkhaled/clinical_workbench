import shutil
import pydicom.data

src = pydicom.data.get_testdata_file("CT_small.dcm")
shutil.copy(src, "08_CT_small.dcm")

print("Created: 08_CT_small.dcm")