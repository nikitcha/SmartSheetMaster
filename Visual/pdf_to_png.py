import subprocess
import os
import traceback
import sys

# Absolute path to Ghostscript executable or command name if Ghostscript is in PATH.
GHOSTSCRIPTCMD = "gswin64"


def gs_pdf_to_png(pdffilepath, resolution):
    if not os.path.isfile(pdffilepath):
        print("'%s' is not a file. Skip." % pdffilepath)
    #pdffiledir = os.path.dirname(pdffilepath)
    #pdffilename = os.path.basename(pdffilepath)
    pdfname, ext = os.path.splitext(pdffilepath)

    try:    
        # Change the "-rXXX" option to set the PNG's resolution.
        # http://ghostscript.com/doc/current/Devices.htm#File_formats
        # For other commandline options see
        # http://ghostscript.com/doc/current/Use.htm#Options
        arglist = [GHOSTSCRIPTCMD,
                  "-q",                     
                  "-dQUIET",                   
                  "-dPARANOIDSAFER",                    
                  "-dBATCH",
                  "-dNOPAUSE",
                  "-dNOPROMPT",                  
                  "-sOutputFile=" + pdfname + "-%03d.png",
                  "-sDEVICE=png16m",                  
                  "-r%s" % resolution,
                  pdffilepath]
        print("Running command:\n%s" % ' '.join(arglist))
        sp = subprocess.Popen(
            args=arglist,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    except OSError:
        sys.exit("Error executing Ghostscript ('%s'). Is it in your PATH?" %
            GHOSTSCRIPTCMD)            
    except:
        print("Error while running Ghostscript subprocess. Traceback:")
        print("Traceback:\n%s"%traceback.format_exc())

    stdout, stderr = sp.communicate()
    print("Ghostscript stdout:\n'%s'" % stdout)
    if stderr:
        print("Ghostscript stderr:\n'%s'" % stderr)
