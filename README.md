第二章<br/>

simpletokenizer.py #这是一个简单的分词器，使用前需要输入词汇表，方法encode根据词汇表将文字变成数字，decode方法反之 <br/>

简单分词器使用.py #simpletokenizer的简单使用，额外包含了将纯文本（小说）转化成词汇表的步骤，在词汇表里增加了unknow和endoftext以应对更多场景<br/>

tiktoken使用 #tiktoken使用了BPE分词器，用法和simpletokenizer类似<br/>

GPTDataset.py #一个数据加载器<br/>

dataLoaders.py #调用了数据加载器和tiktoken分词器，根据各种参数生成（输入 目标）对<br/>

输入目标对原理.py #简单解释了（输入 目标）对想达到的一个效果<br/>

tokenID到嵌入向量转换.py<br/>

第三章<br/>
