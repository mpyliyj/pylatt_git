def proptemp(name,ponly=0):
    a = "    @property\n" + \
        "    def %s(self):\n"%name +\
        "        return self._%s\n"%name
    
    b =  "    @%s.setter\n"%name +\
         "    def %s(self,value):\n"%name +\
         "        try:\n" +\
         "            self._%s = float(value)\n"%name +\
         "            self._update()\n" +\
         "        except:\n" +\
         "            raise RuntimeError('%s must be float (or convertible)')\n"%name
    if ponly:
        return a
    else:
        return a+b
       
