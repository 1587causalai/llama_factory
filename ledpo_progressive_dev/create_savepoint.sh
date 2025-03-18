# 创建Git保存点示例
cd /path/to/your/project
git add .
git commit -m "修复beta head更新机制 - 解决ref_model导致的梯度断链问题"
git tag -a "v0.7.0-beta-head-fix" -m "成功修复beta head更新机制，关键在于修复了ref_model前向传播导致的梯度断链"
echo "保存点创建成功: v0.7.0-beta-head-fix"
echo "如需查看所有保存点:"
echo "  git tag -l -n"
echo "如需回退到此保存点:"
echo "  git checkout v0.7.0-beta-head-fix"
echo "或创建新分支:"
echo "  git checkout -b fix_branch v0.7.0-beta-head-fix"
