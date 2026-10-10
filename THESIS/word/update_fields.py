"""
Opens a .docx in a headless LibreOffice, updates every index (contents,
figures, tables), writes the index texts (entries with page numbers) to
JSON and exports a PDF. The .docx itself is not rewritten.
Run with the system python3 that provides the 'uno' module:

  /usr/bin/python3 THESIS/word/update_fields.py in.docx indexes.json out.pdf
"""
import json
import os
import subprocess
import sys
import time

import uno
from com.sun.star.beans import PropertyValue


def prop(name, value):
    p = PropertyValue()
    p.Name, p.Value = name, value
    return p


def main(src, out_json, out_pdf):
    env = dict(os.environ, HOME=os.environ.get("LO_HOME", "/tmp/claude-0/lohome"))
    proc = subprocess.Popen(["soffice", "--headless", "--norestore", "--nologo", "--nodefault",
                             "--accept=socket,host=127.0.0.1,port=2083;urp;"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        local = uno.getComponentContext()
        resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
        for _ in range(60):
            try:
                ctx = resolver.resolve("uno:socket,host=127.0.0.1,port=2083;urp;StarOffice.ComponentContext")
                break
            except Exception:
                time.sleep(1)
        desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(os.path.abspath(src)), "_blank", 0,
                                           (prop("Hidden", True),))
        for _ in range(2):
            idx = doc.getDocumentIndexes()
            for k in range(idx.getCount()):
                idx.getByIndex(k).update()
            doc.getTextFields().refresh()
            doc.refresh()
        idx = doc.getDocumentIndexes()
        texts = [dict(name=idx.getByIndex(k).getName(), service=idx.getByIndex(k).getImplementationName(),
                      text=idx.getByIndex(k).getAnchor().getString()) for k in range(idx.getCount())]
        with open(out_json, "w") as f:
            json.dump(texts, f, ensure_ascii=False, indent=1)
        doc.storeToURL(uno.systemPathToFileUrl(os.path.abspath(out_pdf)),
                       (prop("FilterName", "writer_pdf_Export"),))
        print("indexes", idx.getCount())
        doc.close(True)
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main(*sys.argv[1:4])
