#pragma once

#include <deque>
#include <string>
#include <vector>

#include "topo.h"

#define MAX_STR_LEN 255
#define MAX_ATTRS   16

namespace mccl {

struct mcclXmlNode {
  char name[MAX_STR_LEN + 1];
  struct {
    char key[MAX_STR_LEN + 1];
    char value[MAX_STR_LEN + 1];
  } attrs[MAX_ATTRS];
  int nAttrs;
  mcclXmlNode* parent;
  std::vector<mcclXmlNode*> subs;
};

struct mcclXml {
  std::deque<mcclXmlNode> nodes;
};

mcclXmlNode* mcclXmlAddNode(mcclXml* xml, mcclXmlNode* parent, const char* name);
void         mcclXmlSetAttr(mcclXmlNode* node, const char* key, const char* value);
const char*  mcclXmlGetAttr(const mcclXmlNode* node, const char* key);
std::string  mcclXmlToString(const mcclXml* xml);

void mcclTopoToXml(const mcclTopoSystem& sys, mcclXml* xml);

mcclResult mcclTopoDumpXmlToFile(const char* path, const mcclXml* xml);
mcclResult mcclTopoFuseXml(mcclXml* dst, const mcclXml* src);

}
