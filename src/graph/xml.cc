#include "xml.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

namespace mccl {

namespace {

void setStr(char* dst, const char* src) {
  std::snprintf(dst, MAX_STR_LEN + 1, "%s", src ? src : "");
}

std::string fmtF(float v) {
  char b[32];
  std::snprintf(b, sizeof(b), "%.3g", static_cast<double>(v));
  return b;
}

std::string xmlEscape(const char* s) {
  std::string out;
  for (const char* p = s; *p; ++p) {
    switch (*p) {
      case '&':  out += "&amp;";  break;
      case '<':  out += "&lt;";   break;
      case '>':  out += "&gt;";   break;
      case '"':  out += "&quot;"; break;
      default:   out += *p;       break;
    }
  }
  return out;
}

void emit(const mcclXmlNode* n, int depth, std::ostringstream& os) {
  const std::string indent(static_cast<size_t>(depth) * 2, ' ');
  os << indent << "<" << n->name;
  for (int i = 0; i < n->nAttrs; ++i)
    os << " " << n->attrs[i].key << "=\"" << xmlEscape(n->attrs[i].value) << "\"";
  if (n->subs.empty()) { os << "/>\n"; return; }
  os << ">\n";
  for (const mcclXmlNode* s : n->subs) emit(s, depth + 1, os);
  os << indent << "</" << n->name << ">\n";
}

}

mcclXmlNode* mcclXmlAddNode(mcclXml* xml, mcclXmlNode* parent, const char* name) {
  xml->nodes.emplace_back();
  mcclXmlNode* n = &xml->nodes.back();
  setStr(n->name, name);
  n->nAttrs = 0;
  n->parent = parent;
  n->subs.clear();
  if (parent != nullptr) parent->subs.push_back(n);
  return n;
}

void mcclXmlSetAttr(mcclXmlNode* node, const char* key, const char* value) {
  for (int i = 0; i < node->nAttrs; ++i)
    if (std::strcmp(node->attrs[i].key, key) == 0) { setStr(node->attrs[i].value, value); return; }
  if (node->nAttrs >= MAX_ATTRS) return;
  setStr(node->attrs[node->nAttrs].key, key);
  setStr(node->attrs[node->nAttrs].value, value);
  node->nAttrs++;
}

const char* mcclXmlGetAttr(const mcclXmlNode* node, const char* key) {
  for (int i = 0; i < node->nAttrs; ++i)
    if (std::strcmp(node->attrs[i].key, key) == 0) return node->attrs[i].value;
  return nullptr;
}

std::string mcclXmlToString(const mcclXml* xml) {
  std::ostringstream os;
  if (!xml->nodes.empty()) emit(&xml->nodes.front(), 0, os);
  return os.str();
}

void mcclTopoToXml(const mcclTopoSystem& sys, mcclXml* xml) {
  xml->nodes.clear();
  mcclXmlNode* root = mcclXmlAddNode(xml, nullptr, "mcclsystem");
  mcclXmlSetAttr(root, "macs", std::to_string(sys.nodes.size()).c_str());
  mcclXmlSetAttr(root, "lan_all_to_all", sys.lanAllToAll ? "1" : "0");
  for (const mcclTopoNode& nd : sys.nodes) {
    mcclXmlNode* x = mcclXmlAddNode(xml, root, "mac");
    mcclXmlSetAttr(x, "rank", std::to_string(nd.rank).c_str());
    if (!nd.host.empty()) mcclXmlSetAttr(x, "host", nd.host.c_str());
    if (nd.gpuCores > 0)  mcclXmlSetAttr(x, "gpucores", std::to_string(nd.gpuCores).c_str());
    if (nd.umaGiB > 0)    mcclXmlSetAttr(x, "uma_gib", std::to_string(nd.umaGiB).c_str());
    if (nd.chipCap > 0)   mcclXmlSetAttr(x, "chipcap", std::to_string(nd.chipCap).c_str());
    for (const mcclTopoLink& l : nd.links) {
      mcclXmlNode* lk = mcclXmlAddNode(xml, x, "link");
      mcclXmlSetAttr(lk, "type", mcclLinkTypeStr(l.type));
      mcclXmlSetAttr(lk, "peer", std::to_string(l.remote).c_str());
      mcclXmlSetAttr(lk, "bw_gbps", fmtF(l.bw).c_str());
    }
  }
}

namespace {

const char* attrOf(const mcclXmlNode* n, const char* key) {
  for (int i = 0; i < n->nAttrs; ++i)
    if (std::strcmp(n->attrs[i].key, key) == 0) return n->attrs[i].value;
  return nullptr;
}

bool nodesMatch(const mcclXmlNode* a, const mcclXmlNode* b) {
  if (std::strcmp(a->name, b->name) != 0) return false;
  const char* key = attrOf(b, "rank") ? "rank" : attrOf(b, "peer") ? "peer" : nullptr;
  if (key == nullptr) return true;
  const char* av = attrOf(a, key);
  const char* bv = attrOf(b, key);
  return av != nullptr && bv != nullptr && std::strcmp(av, bv) == 0;
}

mcclResult addTree(mcclXml* dst, mcclXmlNode* dstParent, const mcclXmlNode* src) {
  mcclXmlNode* n = mcclXmlAddNode(dst, dstParent, src->name);
  for (int i = 0; i < src->nAttrs; ++i) mcclXmlSetAttr(n, src->attrs[i].key, src->attrs[i].value);
  for (const mcclXmlNode* s : src->subs) { mcclResult r = addTree(dst, n, s); if (r != mcclSuccess) return r; }
  return mcclSuccess;
}

mcclResult fuseRec(mcclXml* dst, mcclXmlNode* dstParent, const mcclXmlNode* srcParent) {
  for (const mcclXmlNode* srcNode : srcParent->subs) {
    mcclXmlNode* dstNode = nullptr;
    for (mcclXmlNode* cand : dstParent->subs)
      if (nodesMatch(cand, srcNode)) { dstNode = cand; break; }
    mcclResult r = (dstNode == nullptr) ? addTree(dst, dstParent, srcNode) : fuseRec(dst, dstNode, srcNode);
    if (r != mcclSuccess) return r;
  }
  return mcclSuccess;
}

}

mcclResult mcclTopoDumpXmlToFile(const char* path, const mcclXml* xml) {
  if (path == nullptr || xml == nullptr) return mcclInvalidArgument;
  FILE* file = std::fopen(path, "w");
  if (file == nullptr) { std::fprintf(stderr, "mccl: unable to open %s, not dumping topology\n", path); return mcclSuccess; }
  const std::string s = mcclXmlToString(xml);
  std::fwrite(s.data(), 1, s.size(), file);
  std::fclose(file);
  return mcclSuccess;
}

mcclResult mcclTopoFuseXml(mcclXml* dst, const mcclXml* src) {
  if (dst == nullptr || src == nullptr) return mcclInvalidArgument;
  if (src->nodes.empty()) return mcclSuccess;
  if (dst->nodes.empty()) return addTree(dst, nullptr, &src->nodes.front());
  return fuseRec(dst, &dst->nodes.front(), &src->nodes.front());
}

}

#ifdef MCCL_XML_MAIN
#include "discover.h"
#include <cstdlib>

int main() {
  using namespace mccl;
  const char* nv = std::getenv("MCCL_TOPO_NODES");
  const int n = nv ? std::atoi(nv) : 3;
  std::vector<mcclEdge> edges;
  std::vector<uint32_t> lanIp(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) lanIp[static_cast<size_t>(i)] = 0x0A000000u + static_cast<uint32_t>(i + 1);
  for (int i = 1; i < n; ++i) {
    mcclEdge e;
    e.a = 0; e.b = i; e.live = true;
    e.ipA = 0xC0A80000u + static_cast<uint32_t>(i); e.ipB = 0xC0A80100u + static_cast<uint32_t>(i);
    edges.push_back(e);
  }
  mcclTopoSystem sys;
  mcclTopoGetSystem(n, edges.data(), static_cast<int>(edges.size()), lanIp.data(), 4, &sys);
  mcclTopoComputePaths(&sys);
  mcclXml xml;
  mcclTopoToXml(sys, &xml);
  if (const char* dump = std::getenv("MCCL_TOPO_DUMP_FILE")) mcclTopoDumpXmlToFile(dump, &xml);
  std::fputs(mcclXmlToString(&xml).c_str(), stdout);
  return 0;
}
#endif
