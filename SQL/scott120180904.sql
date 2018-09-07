#1. 사원 테이블에서 이름에 A를 포함하고 커미션을 받는 사원의 사원번호,이름,급여,커미션 출력
desc emp;
select * from emp;

select empno,ename,sal,comm
from emp
where ename like '%A%'
and comm is not null;

#1981년에 입사한 모든 사원의 이름과 입사일을 구하는 질의문은?, 사원명순으로 정렬
select ename,hiredate
from emp
where hiredate like '81%'
order by ename;

#보너스가 급여의 20%이상이고 부서번호가 30인 많은 모든 사원에 대해서 이름,급여 그리고 보너스를 출력하는 질의문 형성하라.
select ename,sal,comm
from emp
where comm>=sal*0.2
and deptno=30;
