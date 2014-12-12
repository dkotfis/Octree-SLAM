//OBJCORE- A Obj Mesh Library by Yining Karl Li
//This file is part of OBJCORE, Coyright (c) 2012 Yining Karl Li

#ifndef OBJLOADER_H
#define OBJLOADER_H

#include <stdlib.h>
#include "obj.h"

using namespace std;

class objLoader{
private:
	obj* geomesh;
public:
	objLoader(string, obj*);
	~objLoader();
    
    //------------------------
    //-------GETTERS----------
    //------------------------
    
	obj* getMesh();
};

#endif