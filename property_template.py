def proptemp(name):
    a  = "    @property\n" + \
         "    def %s():\n"%name +\
         "        return self._%s\n\n"%name +\
         "    @%s.setter\n"%name +\
         "    def %s(self,value):\n"%name +\
         "        try:\n" +\
         "            self.%s = float(value)\n"%name +\
         "            self.update()\n" +\
         "        except:\n" +\
         "            raise RuntimeError('%s must be float (or convertible)')\n"%name
    return a
       
