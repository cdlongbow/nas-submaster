# 版本管理规范

## 1. 分支策略

### 分支结构
```
main          # 主分支，稳定版本，随时可发布
├── dev       # 开发分支，整合所有功能开发
├── feat/*    # 功能分支，从 dev 创建
├── fix/*     # 修复分支，从 dev 创建
└── hotfix/*  # 紧急修复分支，从 main 创建
```

### 分支命名规范
```bash
# 功能分支
feat/<功能简述>
例：feat/subtitle-translation, feat/media-scanner-optimization

# 修复分支
fix/<问题简述>
例：fix/xss-vulnerability, fix/database-connection-leak

# 热修复分支
hotfix/<问题简述>
例：hotfix/security-bypass
```

### 工作流程
```
1. 从 dev 创建功能/修复分支
2. 在分支上开发测试
3. 提交 PR 到 dev，进行代码 review
4. 合并后删除分支
5. 稳定版本从 dev 合并到 main
```

## 2. 提交信息规范

### Commit Message 格式
```
<类型>(<范围>): <简短描述>

[可选正文]

[可选页脚]
```

### 类型标识
| 类型 | 说明 |
|------|------|
| feat | 新功能 |
| fix | 缺陷修复 |
| docs | 文档更新 |
| style | 代码格式（不影响功能） |
| refactor | 重构（既不修复也不添加功能） |
| perf | 性能优化 |
| test | 测试相关 |
| chore | 构建/工具相关 |

### 示例
```bash
# 简单修复
fix(whisper): 修复模型卸载后无法重新加载的问题

# 功能提交
feat(scanner): 添加3级子目录选择器

# 带正文和关联issue的提交
fix(translator): 修复翻译重复问题

修复了当字幕行数超过500时出现的重复翻译bug。
引入的原因是批次大小设置过大导致上下文丢失。

Closes #123
```

## 3. 版本号规范

### 格式
```
<主版本>.<次版本>.<修订版本>[-预发布标签]
简称 X.Y.Z

例：
1.0.0   — 初始正式版本
1.2.3   — 第三个修订版本，第二个次版本
2.0.0   — 重大更新，不兼容变更
1.3.0-beta.1  — 预发布版本
2.1.0-rc.2    — 第二个发布候选版本
```

### 版本号含义

| 版本层级 | 变更类型 | 示例 | 说明 |
|---------|---------|------|------|
| **主版本 (X)** | 破坏性变更 | 1.0.0 → 2.0.0 | 不兼容的API或架构调整 |
| **次版本 (Y)** | 新功能 | 1.2.0 → 1.3.0 | 向后兼容的功能新增 |
| **修订版本 (Z)** | 问题修复 | 1.2.3 → 1.2.4 | 向后兼容的缺陷修复 |

### 版本号递增规则
```bash
# 主版本递增：做了不兼容的改动
1.0.0 → 2.0.0
例：重构核心数据模型、更改数据库表结构、改变API接口

# 次版本递增：添加了向后兼容的新功能
1.0.0 → 1.1.0
例：新增3级目录选择器、新增字幕格式转换功能

# 修订版本递增：修复了向后兼容的bug
1.0.1 → 1.0.2
例：修复内存泄漏、修复数据库连接问题、修复XSS漏洞
```

### 预发布版本标签
```bash
-alpha.1   # 开发中版本，可能不稳定
-beta.2    # 测试版本，功能基本完整
-rc.1      # 候选发布版本(Release Candidate)，接近正式发布
```

### 版本号示例（对应本项目）
```bash
v1.0.0   — 基础字幕提取功能
v1.1.0   — 新增LLM翻译功能
v1.1.1   — 修复翻译模块的内存泄漏
v1.2.0   — 新增多格式导出（VTT/ASS）
v2.0.0   — 重构UI框架，破坏性变更
```

### 标签命名
```bash
# Release 标签
v1.0.0
v1.2.3

# 预发布标签
v1.3.0-beta.1
v2.0.0-rc.1
```

## 4. 发布流程

### 正式发布
```bash
# 1. 确保 dev 分支通过所有测试
# 2. 更新版本号（在代码中）
git checkout dev
git pull

# 3. 合并到 main
git checkout main
git pull
git merge dev
git tag -a v1.x.x -m "Release v1.x.x"
git push origin main --tags

# 4. 合并 main 变更回 dev
git checkout dev
git merge main
git push origin dev
```

### 快速修复发布
```bash
# 从 main 创建热修复分支
git checkout main
git checkout -b hotfix/fix-description

# 修复后直接合并到 main
git checkout main
git merge hotfix/fix-description
git tag -a v1.x.x -m "Hotfix v1.x.x"
git push origin main --tags

# 合并回 dev
git checkout dev
git merge main
git push origin dev

# 删除热修复分支
git branch -d hotfix/fix-description
```

## 5. Git 配置建议

### 推荐的全局配置
```bash
# 合并工具（解决冲突）
git config --global merge.tool vimdiff

# 默认分支
git config --global init.defaultBranch main

# 拉取策略
git config --global pull.rebase false

# 提交模板（可选）
git config --global commit.template ~/.gitmessage
```

### 推荐的提交模板 (~/.gitmessage)
```
#类型(范围): 简短描述

# 详细描述（可选）

#-------------------
# 类型: feat | fix | docs | style | refactor | perf | test | chore
# 范围: whisper | translator | scanner | ui | database | core | ...
# 描述: 动词开头，不超过50字符
```

## 6. 注意事项

- **不要提交敏感信息**（API密钥、密码等）到仓库
- **保持提交原子性**：一个提交只做一件事
- **频繁提交**：不要等到开发完成才提交，早提交早备份
- **写清楚提交信息**：方便 code review 和追溯问题
- **及时同步**：开始工作前先 pull，结束后及时 push
