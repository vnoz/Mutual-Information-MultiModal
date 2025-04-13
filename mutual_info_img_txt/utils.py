from matplotlib import pyplot as plt

class MimicID:
	subject_id = ''
	study_id = ''
	dicom_id = ''

	def __init__(self, subject_id, study_id, dicom_id):
		self.subject_id = str(subject_id)
		self.study_id = str(study_id)
		self.dicom_id = str(dicom_id)

	def __str__(self):
		return f"p{self.subject_id}_s{self.study_id}_{self.dicom_id}"

	@staticmethod
	def get_study_id(mimic_id: str):
		return mimic_id.split('_')[1][1:]


def PrintModel(model):
	for name, param in model.named_parameters():
		if 'weight' in name:
			print(f"Layer: {name}, Shape: {param.shape}")
			
			print(param.data)


def Plot_Training(xlabel, ylabel, title, data, dataLabel,out_imgage_file):
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
		
	for i in range(len(data)):
		plt.plot(data[i],label=dataLabel[i])
	
	plt.legend()
	plt.show()
	plt.savefig(out_imgage_file)
	plt.clf()

def Plot_Training_From_Logfile(logFile):
	#read log file and find keywords: training loss, validation loss, training accuracy and get loss and accuracy arrays, then call Plot_Training
	i=0