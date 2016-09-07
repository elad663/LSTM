require 'nngraph'
x1=torch.rand(4)
x2=torch.rand(4)
x3=torch.rand(4)


n1=nn.Linear(4,4)()
n2=nn.Identity()()
n3=nn.CMulTable()({n1,n2})
n4=nn.Identity()()
n5=nn.CAddTable()({n4,n3})

m=nn.gModule({n1,n2,n4},{n5})
output=m:forward({x1,x2,x3})

