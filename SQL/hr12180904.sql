#1. �л� ���̺��� �̸��� '��'���γ�����,���������� ���������ʴ� 101�� �а� �л��� ���̵�,�̸�,�г�,�а� ��ȣ�� ���
desc student;
desc professor;
select * from student;
select * from professor;
select userid,name,grade,deptno
from student
where name like '%��'
and profno is null
and deptno=101;
#3. �л� ���̺�� �������̺� ���� union������ �����Ͽ� name,userid,sal�� ���, sal������ 0��� 
select name,userid,0
from student
union
select name,userid,sal
from professor;
#4. ���� ���̺��� ��ü ������ �޿��� �λ��ϱ� ���� ���� ����� ���(�� ������ ���Ӱ����� ������� ��ܿ��� ����)
select name,position
from professor
where not position='���Ӱ���';

select name,position
from professor
minus
select name,position
from professor
where position='���Ӱ���';

#4. ���� ���̺��� ������ȣ,�̸�,����,�޿� �׸��� 19%������ �޿��� �̸�,���������� ���(�����ȱ޿��� �� ���̺��� newsalary)
select profno,name,position,sal,sal*1.19 New_Salary
from professor
order by name,position;

#������̺��� 'scott'����� �̸�,�μ���ȣ,�޿��� ���
select ename,deptno,sal
from emp
where INITCAP(ename)='scott';

select empno,ename,job
from emp
where lower(job)='manager';

#�̸��� A,T,M���� �����ϴ� ��� ����� ���ؼ� ù��°���ڴ� �빮�� �������� �ҹ��ڷ� ��Ÿ���� ����� �̸��� �̸����̸� ���
select initcap(ename),Length(ename) "Length"
from emp
where ename like 'A%'
or ename like 'T%'
or ename like 'M%';

#1. �������� ������� ��ĥ°���� ���
select to_date(19950924,'yyyy/mm/dd')
from dual;
select trunc(sysdate - to_date(19950924,'yyyy/mm/dd')) "lived day"
from dual;


#1. ����(job)�� manager�̰ų� salesman�̸� �޿��� $1500,3000,5000�� �ƴ� ��� ����� ���ؼ� �̸� ���� �׸��� �޿������
select * from emp;
select ename,job,sal
from emp
where job in ('MANAGER','SALESMAN')
and sal not in(1500,3000,5000);
#2. ��� ���̺��� �����ȣ,�̸�,����,�޿� �׸��� 22%������ �޿��� �̸�,���������� ���
select empno,ename,job,sal,sal*1.22 "New Salary"
from emp
order by ename ,job;
#3. 2�� �߰��Ͽ� ���ο� �޿����� ������ �޿�(sal)�� ���� ���� �߰�
select empno,ename,job,sal,sal*1.22 "New Salary",sal*1.22-sal Increase
from emp
order by ename ,job;
#4. ����� �̸��� ���ʽ��� ����ϴ� ���Ǹ� �ۼ�. ���ʽ��ȹ����� no commission���
select ename,NVL(to_char(comm),'No Commission') comm
from emp;
