import sys
import os.path,subprocess
from subprocess import STDOUT,PIPE

# compile
subprocess.check_call(['javac', 'C:/ITi/workspace_web/web_prj/src/com/test/JavaModule.java'])

# execute
cmd = ['java', 'JavaModule']
proc = subprocess.Popen(cmd, stdout=PIPE, stderr = STDOUT)
stdout,stderr = proc.communicate()
print(stdout)
