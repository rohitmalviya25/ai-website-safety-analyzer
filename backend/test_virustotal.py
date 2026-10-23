
from virustotal_check import check_virustotal

# A public example URL for testing an existing report.
result = check_virustotal("https://www.google.com/")
print(result)
